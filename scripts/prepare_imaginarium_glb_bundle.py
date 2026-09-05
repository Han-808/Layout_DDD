#!/usr/bin/env python3
"""Plan, optionally build, and verify a selected Imaginarium GLB bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmark.generation_comparison.imaginarium_bundle import (
    GEOMETRY_TOLERANCE_M,
    build_imaginarium_glb_bundle_plan,
    validate_imaginarium_glb_bundle,
)
from benchmark.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--blender-executable", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()
    if args.tolerance != GEOMETRY_TOLERANCE_M:
        raise ValueError(
            f"bundle tolerance is frozen at {GEOMETRY_TOLERANCE_M:g} metres"
        )

    spec = read_json(args.spec)
    catalog = spec.get("catalog") if isinstance(spec, dict) else None
    if not isinstance(catalog, dict):
        raise ValueError("comparison spec must contain a catalog object")
    plan = build_imaginarium_glb_bundle_plan(
        catalog_spec=catalog,
        asset_root=args.asset_root,
        bundle_root=args.bundle_root,
    )
    result: dict[str, object] = {
        "plan": plan["plan_path"],
        "asset_count": plan["asset_count"],
        "blender_executed": False,
    }
    if args.blender_executable is not None:
        worker = args.worker or (
            Path(__file__).resolve().parent
            / "blender"
            / "convert_imaginarium_frozen_bundle.py"
        )
        report = args.bundle_root / "bundle_report.json"
        completed = subprocess.run(
            [
                args.blender_executable.expanduser().resolve().as_posix(),
                "--background",
                "--python",
                worker.expanduser().resolve().as_posix(),
                "--",
                "--plan",
                str(plan["plan_path"]),
                "--report",
                report.resolve().as_posix(),
                "--tolerance",
                str(args.tolerance),
            ],
            check=False,
        )
        result["blender_executed"] = True
        result["blender_return_code"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(
                f"Blender asset conversion failed with code {completed.returncode}"
            )
        validation = validate_imaginarium_glb_bundle(
            plan=plan["plan_path"],
            report=report,
            expected_asset_root=args.asset_root,
            expected_bundle_root=args.bundle_root,
        )
        result["report"] = report.resolve().as_posix()
        result["validation"] = validation
        if not validation["valid"]:
            raise RuntimeError("converted Imaginarium GLB bundle failed validation")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
