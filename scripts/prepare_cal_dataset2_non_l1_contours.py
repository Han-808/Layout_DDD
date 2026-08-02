#!/usr/bin/env python3
"""Prepare frozen same-pose contour evidence for cal_dataset2 OOR/OAR.

The existing review render bank already contains real-mesh global and local RGB
views.  This preparation step re-materializes only the OOR/OAR scenes in one
Blender process per observation, renders visible target-ID masks at the frozen
local camera poses, and composes the repository's segmentation-contour
presentation.  It never reads semantic GT or construction proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering.segmentation_contour import (  # noqa: E402
    compose_segmentation_contour_manifest,
)


DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
RENDER_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_review_renders"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_contours"
)
ASSET_ROOT = PROJECT_ROOT / "Support" / "Assets" / "imaginarium_assets"
BLENDER_BIN = Path("/Applications/Blender.app/Contents/MacOS/Blender")
METRICS = ("oor", "oar")
PALETTE = (
    [235, 74, 78],
    [62, 131, 246],
    [241, 181, 55],
    [116, 201, 115],
)
SCHEMA_VERSION = "cal_dataset2_non_l1_contour_evidence_v1"


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    render_root = args.render_root.expanduser().resolve()
    output_root = args.out_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    _preflight(dataset_root, render_root, asset_root, blender_bin)

    review = _read_json(render_root / "review" / "review_cases.json")
    cards = [
        card
        for card in review.get("cases") or []
        if isinstance(card, dict)
        and str(card.get("metric")) in METRICS
        and (not args.case_id or str(card.get("case_id")) in set(args.case_id))
    ]
    if not cards:
        raise ValueError("no OOR/OAR review cases matched the requested filters")
    observations: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for card in cards:
        observations.setdefault(str(card["observation_id"]), card)

    worker = (
        PROJECT_ROOT
        / "Support"
        / "scripts"
        / "blender_cal_dataset2_review_worker.py"
    )
    compositor = (
        PROJECT_ROOT / "src" / "benchmark" / "rendering" / "segmentation_contour.py"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.time()
    for index, (observation_id, card) in enumerate(observations.items(), start=1):
        contract = _contract(
            dataset_root=dataset_root,
            render_root=render_root,
            card=card,
            worker=worker,
            compositor=compositor,
            args=args,
        )
        contract_sha256 = _json_sha256(contract)
        destination = (
            output_root
            / "observations"
            / observation_id
            / contract_sha256[:16]
        )
        manifest_path = destination / "evidence_manifest.json"
        if args.resume and _cached_manifest_ready(manifest_path, contract_sha256):
            manifest = _read_json(manifest_path)
            records.append(_index_record(manifest_path, manifest))
            print(
                f"[{index}/{len(observations)}] cached {observation_id} "
                f"{card['metric']} {card['case_id']}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(observations)}] prepare {observation_id} "
            f"{card['metric']} {card['case_id']}",
            flush=True,
        )
        if args.plan_only:
            records.append(
                {
                    "observation_id": observation_id,
                    "metric": card["metric"],
                    "case_id": card["case_id"],
                    "contract_sha256": contract_sha256,
                    "manifest_path": str(manifest_path),
                    "status": "planned",
                }
            )
            continue
        try:
            manifest = _prepare_one(
                dataset_root=dataset_root,
                render_root=render_root,
                output_dir=destination,
                card=card,
                contract=contract,
                contract_sha256=contract_sha256,
                blender_bin=blender_bin,
                asset_root=asset_root,
                worker=worker,
                args=args,
            )
            records.append(_index_record(manifest_path, manifest))
        except Exception as exc:
            failure = {
                "observation_id": observation_id,
                "case_id": str(card["case_id"]),
                "metric": str(card["metric"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[failed] {json.dumps(failure, ensure_ascii=False)}", flush=True)
            if args.fail_fast:
                raise

    summary = {
        "schema_version": "cal_dataset2_non_l1_contour_run_v1",
        "dataset_root": str(dataset_root),
        "render_root": str(render_root),
        "output_root": str(output_root),
        "selected_case_count": len(cards),
        "unique_observation_count": len(observations),
        "record_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "records": records,
        "elapsed_seconds": time.time() - started,
        "plan_only": bool(args.plan_only),
    }
    manifest_name = "plan_manifest.json" if args.plan_only else "run_manifest.json"
    _write_json(output_root / manifest_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def _prepare_one(
    *,
    dataset_root: Path,
    render_root: Path,
    output_dir: Path,
    card: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    blender_bin: Path,
    asset_root: Path,
    worker: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_id = str(card["observation_id"])
    observation_dir = render_root / "observations" / observation_id
    review_manifest = _read_json(observation_dir / "review_render_manifest.json")
    poses = _read_json_array(observation_dir / "camera_views.json")
    target_ids = [str(item["id"]) for item in card.get("target_objects") or []]
    scene = _read_json(
        dataset_root / "fixtures" / str(card["source_case_id"]) / "generated_scene.json"
    )
    object_by_id = {
        str(item.get("id")): item
        for item in scene.get("objects") or []
        if isinstance(item, dict) and item.get("id") is not None
    }
    targets = []
    for index, target_id in enumerate(target_ids):
        item = object_by_id.get(target_id)
        if item is None:
            raise ValueError(f"target {target_id} is absent from the source scene")
        targets.append(
            {
                "id": target_id,
                "color": PALETTE[index % len(PALETTE)],
            }
        )
    worker_out = output_dir / "mask_worker"
    command = [
        str(blender_bin),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(worker),
        "--",
        "--scene-json",
        str(
            dataset_root
            / "fixtures"
            / str(card["source_case_id"])
            / "generated_scene.json"
        ),
        "--camera-views-json",
        str(observation_dir / "camera_views.json"),
        "--out-dir",
        str(worker_out),
        "--asset-root",
        str(asset_root),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--oar-plane-flags-json",
        json.dumps(review_manifest.get("oar_plane_flags") or {}, separators=(",", ":")),
        "--mask-targets-json",
        json.dumps(targets, separators=(",", ":")),
        "--mask-only",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
    )
    (output_dir / "blender.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / "blender.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise RuntimeError(
            f"Blender mask worker exited with {completed.returncode}: {detail}"
        )
    worker_manifest = _read_json(worker_out / "review_worker_manifest.json")
    mask_manifest = _read_json(
        worker_out / "target_id_masks" / "target_id_mask_manifest.json"
    )
    pose_by_id = {str(pose.get("id")): pose for pose in poses}
    rgb_views = []
    for view in review_manifest.get("local_views") or []:
        view_id = str(view.get("id") or "")
        if view_id not in pose_by_id:
            raise ValueError(f"local RGB view {view_id} has no frozen camera pose")
        rgb_views.append(
            {
                "id": view_id,
                "path": str((observation_dir / str(view["path"])).resolve()),
                "pose": pose_by_id[view_id],
            }
        )
    composed = compose_segmentation_contour_manifest(
        rgb_manifest={"views": rgb_views},
        mask_manifest=mask_manifest,
        overlay_spec={"targets": targets},
        out_dir=output_dir / "contour",
        band_width_px=args.band_width_px,
        outline_width_px=args.outline_width_px,
        band_alpha=args.band_alpha,
        outline_alpha=args.outline_alpha,
    )
    contour_views = composed.get("views") or []
    if len(contour_views) != len(rgb_views):
        raise RuntimeError(
            f"incomplete contour packet: rgb={len(rgb_views)} "
            f"contour={len(contour_views)}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": observation_id,
        "representative_case_id": str(card["case_id"]),
        "metric": str(card["metric"]),
        "target_ids": target_ids,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "source_review_manifest_sha256": _file_sha256(
            observation_dir / "review_render_manifest.json"
        ),
        "mask_manifest": str(
            (worker_out / "target_id_masks" / "target_id_mask_manifest.json").resolve()
        ),
        "contour_manifest": str(
            (output_dir / "contour" / "segmentation_contour_manifest.json").resolve()
        ),
        "contour_views": [
            {
                "view_id": str(view["view_id"]),
                "path": str(Path(str(view["output_path"])).resolve()),
                "sha256": _file_sha256(Path(str(view["output_path"]))),
                "target_visible_pixels": {
                    str(target["id"]): int(
                        target.get("visible_pixels_at_composite_resolution") or 0
                    )
                    for target in view.get("targets") or []
                },
            }
            for view in contour_views
        ],
        "local_room_repairs": worker_manifest.get("local_room_repairs") or [],
        "complete": True,
    }
    _write_json(output_dir / "evidence_manifest.json", manifest)
    return manifest


def _contract(
    *,
    dataset_root: Path,
    render_root: Path,
    card: dict[str, Any],
    worker: Path,
    compositor: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    observation_dir = render_root / "observations" / str(card["observation_id"])
    scene_path = (
        dataset_root
        / "fixtures"
        / str(card["source_case_id"])
        / "generated_scene.json"
    )
    return {
        "schema_version": "cal_dataset2_non_l1_contour_contract_v1",
        "observation_id": str(card["observation_id"]),
        "metric": str(card["metric"]),
        "target_ids": [str(item["id"]) for item in card.get("target_objects") or []],
        "source_sha256": {
            "scene": _file_sha256(scene_path),
            "review_manifest": _file_sha256(
                observation_dir / "review_render_manifest.json"
            ),
            "camera_views": _file_sha256(observation_dir / "camera_views.json"),
            "worker": _file_sha256(worker),
            "compositor": _file_sha256(compositor),
        },
        "presentation": {
            "band_width_px": args.band_width_px,
            "outline_width_px": args.outline_width_px,
            "band_alpha": args.band_alpha,
            "outline_alpha": args.outline_alpha,
            "identity_mask": "visible_2d_segmentation_respect_occlusion",
        },
        "semantic_gt_visibility": "never_read",
    }


def _cached_manifest_ready(path: Path, contract_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract_sha256") != contract_sha256
            or manifest.get("complete") is not True
        ):
            return False
        views = manifest.get("contour_views") or []
        return len(views) == 3 and all(
            Path(str(view.get("path") or "")).is_file()
            and _file_sha256(Path(str(view["path"]))) == view.get("sha256")
            for view in views
        )
    except Exception:
        return False


def _index_record(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": manifest["observation_id"],
        "case_id": manifest["representative_case_id"],
        "metric": manifest["metric"],
        "contract_sha256": manifest["contract_sha256"],
        "manifest_path": str(path.resolve()),
        "status": "ready",
    }


def _preflight(
    dataset_root: Path,
    render_root: Path,
    asset_root: Path,
    blender_bin: Path,
) -> None:
    required = (
        dataset_root / "dataset_manifest.json",
        render_root / "review" / "review_cases.json",
        asset_root / "imaginarium_asset_info.csv",
        blender_bin,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"contour preparation preflight missing: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=BLENDER_BIN)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--band-width-px", type=int, default=7)
    parser.add_argument("--outline-width-px", type=int, default=2)
    parser.add_argument("--band-alpha", type=float, default=0.30)
    parser.add_argument("--outline-alpha", type=float, default=0.95)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{path} must contain an array of JSON objects")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
