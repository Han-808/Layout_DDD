#!/usr/bin/env python3
"""Render and serve a blind, local human-review UI for cal_dataset2.

The dataset builder intentionally emits SVG construction previews only.  This
tool materializes the canonical scenes with real Imaginarium meshes, adds
metric/target-local camera views, and creates a static single-case reviewer.

It is local-only and model-free.  It never reads construction proposals or
semantic GT when building the reviewer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.rendering.camera_pose import generate_camera_pose_candidates  # noqa: E402


DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
DEFAULT_OUT_ROOT = (
    REPO_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_review_renders"
)
DEFAULT_ASSET_ROOT = REPO_ROOT / "Support" / "Assets" / "imaginarium_assets"
DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
REVIEW_MANIFEST = "review_render_manifest.json"
MIN_REVIEW_IMAGE_CHANNEL_SPAN = 12.0
MIN_REVIEW_IMAGE_STDDEV = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--phase", choices=("all", "render", "ui"), default="all")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--local-views", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    _preflight(dataset_root, asset_root, blender_bin)

    review_rows = _read_tsv(dataset_root / "review" / "review_queue.tsv")
    inventory_rows = _read_tsv(
        dataset_root / "validation" / "case_inventory.tsv"
    )
    inventory_by_case = {row["case_id"]: row for row in inventory_rows}
    case_ids = [row["case_id"] for row in review_rows]
    selected_case_ids = set(args.case_id or case_ids)
    unknown = sorted(selected_case_ids - set(case_ids))
    if unknown:
        raise ValueError(f"unknown --case-id values: {unknown}")

    observation_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for case_id in case_ids:
        if case_id not in selected_case_ids:
            continue
        spec = _observation_spec(
            dataset_root=dataset_root,
            inventory_by_case=inventory_by_case,
            case_id=case_id,
        )
        observation_id = spec["observation_id"]
        group = observation_groups.setdefault(
            observation_id,
            {
                **spec,
                "member_case_ids": [],
            },
        )
        group["member_case_ids"].append(case_id)
    if args.limit is not None:
        observation_groups = OrderedDict(
            list(observation_groups.items())[: max(0, args.limit)]
        )

    failures: list[dict[str, str]] = []
    rendered = 0
    cached = 0
    if args.phase in {"all", "render"}:
        _migrate_legacy_geometry_cache(
            dataset_root=dataset_root,
            out_root=out_root,
            inventory_by_case=inventory_by_case,
            observation_groups=observation_groups,
        )
        total = len(observation_groups)
        for index, (observation_id, group) in enumerate(
            observation_groups.items(), start=1
        ):
            source_id = str(group["source_case_id"])
            member_ids = list(group["member_case_ids"])
            representative = member_ids[0]
            destination = out_root / "observations" / observation_id
            if _observation_complete(destination):
                cached += 1
                print(
                    f"[{index}/{total}] cached {observation_id} "
                    f"scene={source_id} metric={group['metric']} "
                    f"cases={len(member_ids)}",
                    flush=True,
                )
                continue
            print(
                f"[{index}/{total}] render {observation_id} scene={source_id} "
                f"metric={group['metric']} representative={representative} "
                f"cases={len(member_ids)}",
                flush=True,
            )
            try:
                _render_observation(
                    dataset_root=dataset_root,
                    destination=destination,
                    observation_id=observation_id,
                    source_id=source_id,
                    representative_case_id=representative,
                    member_case_ids=member_ids,
                    blender_bin=blender_bin,
                    asset_root=asset_root,
                    local_view_count=max(1, min(6, args.local_views)),
                    width=max(128, args.width),
                    height=max(128, args.height),
                    timeout_seconds=max(1, args.timeout_seconds),
                )
                rendered += 1
            except Exception as exc:  # keep later cases auditable
                failure = {
                    "observation_id": observation_id,
                    "source_id": source_id,
                    "representative_case_id": representative,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(f"[failed] {json.dumps(failure, ensure_ascii=False)}", flush=True)
                if args.fail_fast:
                    raise

    ui_path = _build_review_ui(
        dataset_root=dataset_root,
        out_root=out_root,
        review_rows=review_rows,
        inventory_by_case=inventory_by_case,
    )
    summary = {
        "schema_version": "cal_dataset2_render_review_summary_v1",
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "review_case_count": len(review_rows),
        "selected_unique_observation_count": len(observation_groups),
        "rendered_observation_count": rendered,
        "cached_observation_count": cached,
        "failure_count": len(failures),
        "failures": failures,
        "review_index": str(ui_path),
        "generated_at_unix": time.time(),
    }
    _write_json(out_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


def _preflight(dataset_root: Path, asset_root: Path, blender_bin: Path) -> None:
    required = (
        dataset_root / "dataset_manifest.json",
        dataset_root / "review" / "review_queue.tsv",
        dataset_root / "validation" / "case_inventory.tsv",
        asset_root / "imaginarium_asset_info.csv",
        blender_bin,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"review render preflight missing files: {missing}")


def _observation_spec(
    *,
    dataset_root: Path,
    inventory_by_case: dict[str, dict[str, str]],
    case_id: str,
) -> dict[str, Any]:
    inventory = inventory_by_case[case_id]
    source_id = inventory.get("visual_source_case_id") or case_id
    fixture = dataset_root / "fixtures" / case_id
    event_payload = _read_json(fixture / "metric_events.json")
    events = event_payload.get("events") or []
    if len(events) != 1 or not isinstance(events[0], dict):
        raise ValueError(f"{case_id} must contain exactly one metric event")
    event = events[0]
    metric = str(event.get("metric") or "")
    target_ids = [
        str(value) for value in event.get("target_ids") or [] if str(value)
    ]
    plane_flags = (
        _oar_plane_flags(_read_json(fixture / "specification_contract.json"))
        if metric == "oar"
        else {}
    )
    signature = {
        "source_case_id": source_id,
        "metric": metric,
        "target_ids": target_ids,
        "plane_flags": plane_flags,
    }
    if metric == "oar":
        # Architecture-plane-normal views can place an opaque wall between the
        # camera and the attached target.  OAR human review instead uses
        # target-centred oblique views while retaining global room context.
        # Version this only for OAR so existing non-OAR renders remain reusable.
        signature["oar_local_camera_mode"] = (
            "target_center_oblique_wall_shell_outside_v3"
        )
    encoded = json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **signature,
        "observation_id": f"obs_{hashlib.sha256(encoded).hexdigest()[:16]}",
    }


def _migrate_legacy_geometry_cache(
    *,
    dataset_root: Path,
    out_root: Path,
    inventory_by_case: dict[str, dict[str, str]],
    observation_groups: OrderedDict[str, dict[str, Any]],
) -> None:
    """Recover correctly rendered legacy caches under the stricter key.

    The first renderer keyed caches only by scene geometry.  A complete cache is
    still correct for the representative metric/target request recorded in its
    manifest, so move it to that exact observation key.  Other metrics sharing
    the geometry are rendered separately.
    """

    legacy_root = out_root / "cases"
    if not legacy_root.is_dir():
        return
    observation_root = out_root / "observations"
    observation_root.mkdir(parents=True, exist_ok=True)
    for legacy_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        if not _observation_complete(legacy_dir):
            continue
        manifest_path = legacy_dir / REVIEW_MANIFEST
        manifest = _read_json(manifest_path)
        representative = str(manifest.get("representative_case_id") or "")
        if representative not in inventory_by_case:
            continue
        spec = _observation_spec(
            dataset_root=dataset_root,
            inventory_by_case=inventory_by_case,
            case_id=representative,
        )
        observation_id = spec["observation_id"]
        group = observation_groups.get(observation_id)
        if group is None:
            continue
        destination = observation_root / observation_id
        if destination.exists():
            continue
        shutil.move(str(legacy_dir), str(destination))
        migrated = _read_json(destination / REVIEW_MANIFEST)
        migrated["observation_id"] = observation_id
        migrated["member_case_ids"] = list(group["member_case_ids"])
        migrated["cache_migration"] = {
            "from_geometry_only_key": legacy_dir.name,
            "to_metric_target_observation_key": observation_id,
        }
        _write_json(destination / REVIEW_MANIFEST, migrated)


def _render_observation(
    *,
    dataset_root: Path,
    destination: Path,
    observation_id: str,
    source_id: str,
    representative_case_id: str,
    member_case_ids: list[str],
    blender_bin: Path,
    asset_root: Path,
    local_view_count: int,
    width: int,
    height: int,
    timeout_seconds: int,
) -> None:
    source_fixture = dataset_root / "fixtures" / source_id
    representative_fixture = dataset_root / "fixtures" / representative_case_id
    scene_path = source_fixture / "generated_scene.json"
    if not scene_path.is_file():
        raise FileNotFoundError(f"missing visual-source scene: {scene_path}")
    scene = _read_json(scene_path)
    event_payload = _read_json(representative_fixture / "metric_events.json")
    events = event_payload.get("events") or []
    if len(events) != 1 or not isinstance(events[0], dict):
        raise ValueError(
            f"{representative_case_id} must contain exactly one metric event"
        )
    event = events[0]
    metric = str(event.get("metric") or "")
    target_ids = [
        str(value) for value in event.get("target_ids") or [] if str(value)
    ]
    destination.mkdir(parents=True, exist_ok=True)

    request: dict[str, Any] = {
        "scene": scene,
        "metric": metric,
        "object_ids": target_ids,
        "_camera_render": {
            "width": width,
            "height": height,
        },
    }
    plane_flags = (
        _oar_plane_flags(
            _read_json(representative_fixture / "specification_contract.json")
        )
        if metric == "oar"
        else {}
    )
    candidates = generate_camera_pose_candidates(
        request,
        max_candidates=local_view_count,
        policy="local",
    )
    camera_path = destination / "camera_views.json"
    camera_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    worker = REPO_ROOT / "Support" / "scripts" / "blender_cal_dataset2_review_worker.py"
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
        str(scene_path),
        "--camera-views-json",
        str(camera_path),
        "--out-dir",
        str(destination),
        "--asset-root",
        str(asset_root),
        "--width",
        str(width),
        "--height",
        str(height),
        "--oar-plane-flags-json",
        json.dumps(plane_flags, separators=(",", ":")),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        (destination / "blender.stdout.log").write_text(
            _stream_text(exc.stdout), encoding="utf-8"
        )
        (destination / "blender.stderr.log").write_text(
            _stream_text(exc.stderr), encoding="utf-8"
        )
        raise RuntimeError(
            f"one-process Blender review worker timed out after {timeout_seconds}s"
        ) from exc
    (destination / "blender.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (destination / "blender.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise RuntimeError(
            f"one-process Blender review worker exited with "
            f"{completed.returncode}: {detail}"
        )
    worker_manifest_path = destination / "review_worker_manifest.json"
    if not worker_manifest_path.is_file():
        raise RuntimeError("Blender review worker produced no manifest")
    worker_manifest = _read_json(worker_manifest_path)
    base_views = [
        view for view in worker_manifest.get("base_views") or []
        if isinstance(view, dict)
    ]
    local_views = [
        view for view in worker_manifest.get("local_views") or []
        if isinstance(view, dict)
    ]
    if len(base_views) != 2 or len(local_views) != local_view_count:
        raise RuntimeError(
            f"incomplete review views: base={len(base_views)} "
            f"local={len(local_views)} expected_local={local_view_count}"
        )
    pixel_validation: list[dict[str, Any]] = []
    for view in base_views + local_views:
        path = Path(str(view.get("path") or ""))
        if not path.is_file():
            raise RuntimeError(f"review worker declared missing image {path}")
        quality = _review_image_quality(path)
        if not quality["informative"]:
            raise RuntimeError(
                "review worker produced a near-uniform image: "
                f"{path} span={quality['maximum_channel_span']:.3f} "
                f"stddev={quality['maximum_channel_stddev']:.3f}"
            )
        pixel_validation.append(
            {
                "path": path.relative_to(destination).as_posix(),
                **quality,
            }
        )
    asset_coverage = worker_manifest.get("asset_coverage") or {}
    if int(asset_coverage.get("bbox_proxy_count") or 0) != 0:
        raise RuntimeError(
            f"{source_id} contains bbox proxy objects: {asset_coverage}"
        )
    if int(asset_coverage.get("asset_mesh_count") or 0) <= 0:
        raise RuntimeError(f"{source_id} contains no real asset meshes")

    review_manifest = {
        "schema_version": "cal_dataset2_review_render_v1",
        "observation_id": observation_id,
        "source_case_id": source_id,
        "representative_case_id": representative_case_id,
        "member_case_ids": list(member_case_ids),
        "metric": metric,
        "target_ids": target_ids,
        "scene_geometry_sha256": _scene_geometry_hash(scene),
        "render_backend": "Blender 5.2 Workbench real-asset meshes",
        "asset_coverage": asset_coverage,
        "base_views": [_portable_view(view, destination) for view in base_views],
        "local_views": [_portable_view(view, destination) for view in local_views],
        "local_camera_policy": "local",
        "local_camera_request_metric": request["metric"],
        "oar_local_camera_mode": (
            "target_center_oblique_wall_shell_outside_v3"
            if metric == "oar"
            else None
        ),
        "oar_plane_flags": plane_flags if metric == "oar" else None,
        "local_room_repairs": worker_manifest.get("local_room_repairs") or [],
        "pixel_validation": pixel_validation,
        "heavy_artifacts_retained": False,
        "one_process_render": True,
    }
    _write_json(destination / REVIEW_MANIFEST, review_manifest)
    _remove_heavy_review_intermediates(destination)


def _observation_complete(destination: Path) -> bool:
    manifest_path = destination / REVIEW_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    views = list(manifest.get("base_views") or []) + list(
        manifest.get("local_views") or []
    )
    if len(manifest.get("base_views") or []) < 2:
        return False
    if len(manifest.get("local_views") or []) < 1:
        return False
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("path"), str):
            return False
        path = destination / view["path"]
        if not path.is_file():
            return False
        try:
            quality = _review_image_quality(path)
        except (OSError, ValueError):
            return False
        if not quality["informative"]:
            return False
    return True


def _review_image_quality(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        statistics = ImageStat.Stat(rgb)
        channel_span = max(float(high - low) for low, high in extrema)
        channel_stddev = max(float(value) for value in statistics.stddev)
        informative = (
            channel_span >= MIN_REVIEW_IMAGE_CHANNEL_SPAN
            and channel_stddev >= MIN_REVIEW_IMAGE_STDDEV
        )
        return {
            "width": int(rgb.width),
            "height": int(rgb.height),
            "maximum_channel_span": channel_span,
            "maximum_channel_stddev": channel_stddev,
            "informative": informative,
            "thresholds": {
                "minimum_channel_span": MIN_REVIEW_IMAGE_CHANNEL_SPAN,
                "minimum_channel_stddev": MIN_REVIEW_IMAGE_STDDEV,
            },
        }


def _portable_view(view: dict[str, Any], destination: Path) -> dict[str, Any]:
    raw_path = Path(str(view["path"])).resolve()
    try:
        relative = raw_path.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"render view escapes source output: {raw_path}") from exc
    return {
        "id": view.get("id") or view.get("name") or relative.stem,
        "name": view.get("name") or view.get("id") or relative.stem,
        "path": relative.as_posix(),
        "camera_location": view.get("camera_location"),
        "camera_target": view.get("camera_target"),
    }


def _remove_heavy_review_intermediates(destination: Path) -> None:
    blend = destination / "scene.blend"
    if blend.is_file():
        blend.unlink()
    blend_backup = destination / "scene.blend1"
    if blend_backup.is_file():
        blend_backup.unlink()
    collision_dir = destination / "collision_geometry"
    if collision_dir.is_dir():
        shutil.rmtree(collision_dir)
    collision_manifest = destination / "collision_geometry_manifest.json"
    if collision_manifest.is_file():
        collision_manifest.unlink()


def _oar_plane_flags(contract: dict[str, Any]) -> dict[str, bool]:
    claims = ((contract.get("claims") or {}).get("oar") or [])
    if len(claims) != 1 or not isinstance(claims[0], dict):
        raise ValueError("OAR review case must contain exactly one OAR claim")
    element = str(claims[0].get("architectural_element") or "").lower()
    mapping = {
        "west_wall": "west_oob",
        "east_wall": "east_oob",
        "south_wall": "south_oob",
        "north_wall": "north_oob",
        "floor": "floor_oob",
        "ceiling": "ceiling_oob",
    }
    flag = mapping.get(element)
    if flag is None:
        raise ValueError(f"unsupported OAR architectural_element {element!r}")
    return {flag: True}


def _build_review_ui(
    *,
    dataset_root: Path,
    out_root: Path,
    review_rows: list[dict[str, str]],
    inventory_by_case: dict[str, dict[str, str]],
) -> Path:
    review_root = out_root / "review"
    preview_root = review_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    for row in review_rows:
        case_id = row["case_id"]
        fixture = dataset_root / "fixtures" / case_id
        inventory = inventory_by_case[case_id]
        spec = _observation_spec(
            dataset_root=dataset_root,
            inventory_by_case=inventory_by_case,
            case_id=case_id,
        )
        source_id = spec["source_case_id"]
        observation_id = spec["observation_id"]
        scene = _read_json(fixture / "generated_scene.json")
        event_payload = _read_json(fixture / "metric_events.json")
        event = (event_payload.get("events") or [])[0]
        target_ids = [str(value) for value in event.get("target_ids") or []]
        object_by_id = {
            str(obj.get("id")): obj
            for obj in scene.get("objects") or []
            if isinstance(obj, dict)
        }
        target_text = []
        for object_id in target_ids:
            obj = object_by_id.get(object_id)
            if obj is None:
                target_text.append({"id": object_id, "description": "architecture target"})
            else:
                target_text.append(
                    {
                        "id": object_id,
                        "description": str(
                            obj.get("short_desc")
                            or obj.get("description")
                            or obj.get("category")
                            or "object"
                        ),
                    }
                )
        source_dir = out_root / "observations" / observation_id
        render_manifest_path = source_dir / REVIEW_MANIFEST
        rendered_views: list[dict[str, str]] = []
        render_status = "missing"
        if render_manifest_path.is_file() and _observation_complete(source_dir):
            manifest = _read_json(render_manifest_path)
            render_status = "ready"
            for family, views in (
                ("global", manifest.get("base_views") or []),
                ("local", manifest.get("local_views") or []),
            ):
                for view in views:
                    rendered_views.append(
                        {
                            "family": family,
                            "name": str(view.get("name") or view.get("id")),
                            "src": (
                                f"../observations/{observation_id}/"
                                f"{str(view['path']).replace(' ', '%20')}"
                            ),
                        }
                    )
        source_svg = dataset_root / row["preview"]
        copied_svg = preview_root / f"{case_id}.svg"
        shutil.copy2(source_svg, copied_svg)
        cards.append(
            {
                "case_id": case_id,
                "event_id": row["event_id"],
                "metric": row["metric"],
                "level": row["level"],
                "prompt_granularity": row["prompt_granularity"],
                "prompt": row["prompt"],
                "review_question": row["review_question"],
                "required_visible_facts": row["required_visible_facts"].split(" | "),
                "target_objects": target_text,
                "object_count": int(row["object_count"]),
                "source_case_id": source_id,
                "observation_id": observation_id,
                "render_status": render_status,
                "rendered_views": rendered_views,
                "construction_svg": f"previews/{case_id}.svg",
            }
        )

    index_path = review_root / "index.html"
    index_path.write_text(_review_html(cards), encoding="utf-8")
    _write_json(
        review_root / "review_cases.json",
        {
            "schema_version": "cal_dataset2_blind_render_review_v1",
            "case_count": len(cards),
            "cases": cards,
        },
    )
    return index_path


def _review_html(cards: list[dict[str, Any]]) -> str:
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    metrics = sorted({card["metric"] for card in cards})
    options = "".join(
        f'<option value="{html.escape(metric)}">{html.escape(metric)}</option>'
        for metric in metrics
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cal_dataset2 Blender review</title>
<style>
:root{{--bg:#111418;--panel:#1b2026;--muted:#9ba7b4;--line:#38424d;--accent:#62a9ff;--ok:#51c878;--bad:#ff6b6b;--amb:#e8bb55}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:#eef2f6;font:15px/1.45 system-ui,sans-serif}}
button,select,input,textarea{{font:inherit}} .top{{position:sticky;top:0;z-index:10;background:#15191ed9;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}
button,select{{background:#242b33;color:#eef2f6;border:1px solid #4b5865;border-radius:7px;padding:7px 10px}} button:hover{{border-color:var(--accent)}} .grow{{flex:1}} .progress{{color:var(--muted);min-width:150px;text-align:right}}
main{{max-width:1560px;margin:auto;padding:18px}} .identity{{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}} h1{{font-size:25px;margin:0}} .pill{{background:#26384c;color:#b9dcff;border:1px solid #41668d;padding:3px 8px;border-radius:999px}}
.notice{{color:#d7c58c;background:#2c281b;border:1px solid #62562c;border-radius:8px;padding:10px 12px;margin:12px 0}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,.85fr);gap:16px}} .gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-content:start}}
.view{{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}} .view img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#252a30;cursor:zoom-in}} .caption{{padding:7px 9px;color:#c7d0da}}
.sidebar{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px;align-self:start;position:sticky;top:72px}} h2{{font-size:16px;margin:14px 0 5px;color:#a9cfff}} p{{margin:5px 0}} ul{{margin:5px 0 12px;padding-left:20px}} code{{color:#b6d8ff}} .targets{{display:flex;gap:6px;flex-wrap:wrap}} .target{{background:#252f39;border:1px solid #465769;border-radius:6px;padding:4px 6px}}
.labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:12px 0}} .labels button.active[data-value=valid]{{background:#174a2c;border-color:var(--ok)}} .labels button.active[data-value=invalid]{{background:#551f25;border-color:var(--bad)}} .labels button.active[data-value=ambiguous]{{background:#514019;border-color:var(--amb)}}
.checks label{{display:block;margin:8px 0}} .checks input{{margin-right:7px}} textarea{{width:100%;min-height:100px;background:#11161b;color:white;border:1px solid #4a5662;border-radius:7px;padding:9px;resize:vertical}}
.saved{{color:var(--ok);font-size:13px;min-height:20px}} .missing{{grid-column:1/-1;background:#3d2424;color:#ffd2d2;border:1px solid #7b4444;padding:16px;border-radius:8px}}
.lightbox{{display:none;position:fixed;inset:0;background:#000e;z-index:30;align-items:center;justify-content:center;padding:24px}} .lightbox.open{{display:flex}} .lightbox img{{max-width:96vw;max-height:94vh;object-fit:contain}} .kbd{{color:#9ba7b4;font-size:12px}}
@media(max-width:1000px){{.layout{{grid-template-columns:1fr}}.sidebar{{position:static}}}} @media(max-width:650px){{.gallery{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="top">
  <button id="prev">← Previous</button><button id="next">Next →</button>
  <select id="metric"><option value="all">all metrics</option>{options}</select>
  <select id="caseSelect" class="grow"></select>
  <button id="export">Export TSV</button>
  <span class="progress" id="progress"></span>
</div>
<main id="app"></main>
<div id="lightbox" class="lightbox"><img alt="Expanded review image"></div>
<script>
const CASES={data};
const STORE='cal_dataset2_blender_review_v1';
let answers=JSON.parse(localStorage.getItem(STORE)||'{{}}');
let filtered=CASES.slice(), index=0;
const app=document.getElementById('app'), sel=document.getElementById('caseSelect');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function currentAnswer(id){{return answers[id]||{{human_semantic_label:'',prompt_compatible:false,target_mapping_correct:false,needs_render_check:false,notes:''}}}}
function save(id,patch){{answers[id]={{...currentAnswer(id),...patch}};localStorage.setItem(STORE,JSON.stringify(answers));render(false)}}
function applyFilter(){{const m=document.getElementById('metric').value;const old=filtered[index]?.case_id;filtered=CASES.filter(c=>m==='all'||c.metric===m);index=Math.max(0,filtered.findIndex(c=>c.case_id===old));if(index<0)index=0;rebuildSelect();render()}}
function rebuildSelect(){{sel.innerHTML=filtered.map((c,i)=>`<option value="${{i}}">${{esc(c.case_id)}} · ${{esc(c.metric)}}</option>`).join('');sel.value=String(index)}}
function render(scroll=true){{
 const c=filtered[index]; if(!c){{app.innerHTML='<p>No cases in this filter.</p>';return}}
 const a=currentAnswer(c.case_id); sel.value=String(index);
 const done=filtered.filter(x=>currentAnswer(x.case_id).human_semantic_label).length;
 document.getElementById('progress').textContent=`${{index+1}} / ${{filtered.length}} · labeled ${{done}}`;
 const views=c.rendered_views.map(v=>`<figure class="view"><img src="${{esc(v.src)}}" data-full="${{esc(v.src)}}" alt="${{esc(v.name)}}"><figcaption class="caption">${{esc(v.family)}} · ${{esc(v.name)}}</figcaption></figure>`).join('');
 const missing=c.render_status!=='ready'?'<div class="missing">Blender views are not ready for this visual source yet. Run/resume the render command.</div>':'';
 const facts=c.required_visible_facts.map(x=>`<li>${{esc(x)}}</li>`).join('');
 const targets=c.target_objects.map(x=>`<span class="target"><code>${{esc(x.id)}}</code> · ${{esc(x.description)}}</span>`).join('');
 app.innerHTML=`<div class="identity"><h1>${{esc(c.case_id)}}</h1><span class="pill">${{esc(c.metric)}}</span><span>${{esc(c.level)}} · ${{esc(c.prompt_granularity)}}</span></div>
 <div class="notice"><strong>Blind review.</strong> Construction proposal/delta are hidden. Blender views use real asset meshes; SVG is retained only as the target-ID geometry legend.</div>
 <div class="layout"><section class="gallery">${{missing}}${{views}}<figure class="view"><img src="${{esc(c.construction_svg)}}" data-full="${{esc(c.construction_svg)}}" alt="Target geometry legend"><figcaption class="caption">target-ID geometry legend (SVG, not verdict evidence)</figcaption></figure></section>
 <aside class="sidebar"><h2>Prompt</h2><p>${{esc(c.prompt)}}</p><h2>Metric question</h2><p>${{esc(c.review_question)}}</p>
 <h2>Targets</h2><div class="targets">${{targets}}</div><h2>Required visible facts</h2><ul>${{facts}}</ul>
 <h2>Semantic label</h2><div class="labels">${{['valid','invalid','ambiguous'].map(v=>`<button data-label="${{v}}" data-value="${{v}}" class="${{a.human_semantic_label===v?'active':''}}">${{v}}</button>`).join('')}}</div>
 <div class="checks"><label><input id="promptOK" type="checkbox" ${{a.prompt_compatible?'checked':''}}>Prompt compatible</label><label><input id="targetOK" type="checkbox" ${{a.target_mapping_correct?'checked':''}}>Target mapping correct</label><label><input id="renderNeeded" type="checkbox" ${{a.needs_render_check?'checked':''}}>Needs additional render check</label></div>
 <h2>Notes</h2><textarea id="notes" placeholder="Reason for ambiguous, incompatibility, mapping issue, or additional view...">${{esc(a.notes)}}</textarea><p class="saved">${{a.human_semantic_label?'Saved locally':''}}</p><p class="kbd">Keys: J/→ next · K/← previous · 1 valid · 2 invalid · 3 ambiguous</p></aside></div>`;
 document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(c.case_id,{{human_semantic_label:b.dataset.label}}));
 document.getElementById('promptOK').onchange=e=>save(c.case_id,{{prompt_compatible:e.target.checked}});
 document.getElementById('targetOK').onchange=e=>save(c.case_id,{{target_mapping_correct:e.target.checked}});
 document.getElementById('renderNeeded').onchange=e=>save(c.case_id,{{needs_render_check:e.target.checked}});
 document.getElementById('notes').oninput=e=>{{answers[c.case_id]={{...currentAnswer(c.case_id),notes:e.target.value}};localStorage.setItem(STORE,JSON.stringify(answers))}};
 document.querySelectorAll('.view img').forEach(img=>img.onclick=()=>{{document.querySelector('#lightbox img').src=img.dataset.full;document.getElementById('lightbox').classList.add('open')}});
 if(scroll)window.scrollTo(0,0);
}}
function move(delta){{if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render()}}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('metric').onchange=applyFilter;sel.onchange=e=>{{index=Number(e.target.value);render()}};
document.getElementById('lightbox').onclick=e=>e.currentTarget.classList.remove('open');
document.addEventListener('keydown',e=>{{if(e.target.matches('textarea,input,select'))return;if(e.key==='j'||e.key==='ArrowRight')move(1);if(e.key==='k'||e.key==='ArrowLeft')move(-1);if(['1','2','3'].includes(e.key))save(filtered[index].case_id,{{human_semantic_label:{{'1':'valid','2':'invalid','3':'ambiguous'}}[e.key]}})}});
document.getElementById('export').onclick=()=>{{const fields=['case_id','event_id','metric','human_semantic_label','prompt_compatible','target_mapping_correct','needs_render_check','notes'];const q=s=>'"'+String(s??'').replaceAll('"','""')+'"';const rows=[fields.join('\\t'),...CASES.map(c=>fields.map(f=>q(f in c?c[f]:currentAnswer(c.case_id)[f])).join('\\t'))];const blob=new Blob([rows.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='cal_dataset2_human_review.tsv';a.click();URL.revokeObjectURL(a.href)}};
rebuildSelect();render();
</script></body></html>"""


def _scene_geometry_hash(scene: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(scene))
    normalized.pop("scene_id", None)
    normalized.pop("request_id", None)
    if isinstance(normalized.get("metadata"), dict):
        for key in ("case_id", "request_id"):
            normalized["metadata"].pop(key, None)
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


if __name__ == "__main__":
    main()
