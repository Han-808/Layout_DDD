#!/usr/bin/env python3
"""Render all source and mutation scenes for the L3 mutation experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering import BlenderRenderer
from scripts.build_l3_mutation_dataset import (
    DEFAULT_CONFIG,
    file_sha256,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("all", "sources", "variants"),
        default="all",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    summary = render_dataset(
        config,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
        scope=args.scope,
        start_index=args.start_index,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


def render_dataset(
    config: dict[str, Any],
    *,
    resume: bool,
    continue_on_error: bool,
    scope: str,
    start_index: int,
    limit: int | None,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    dataset_path = output_root / "dataset_manifest.json"
    dataset = _read_json(dataset_path)
    render_config = _object(config.get("render"), "render")
    blender_bin = _path(
        render_config["blender_bin"],
        repo_root=PROJECT_ROOT,
    )
    asset_root = _path(
        render_config["asset_root"],
        repo_root=PROJECT_ROOT,
    )
    renderer = BlenderRenderer(
        blender_bin=blender_bin,
        timeout_seconds=int(render_config["timeout_seconds"]),
        width=int(render_config["width"]),
        height=int(render_config["height"]),
        render_engine=str(render_config["engine"]),
        cycles_device=str(render_config.get("cycles_device", "CPU")),
        cycles_samples=int(render_config.get("cycles_samples", 1)),
        cycles_denoising=bool(
            render_config.get("cycles_denoising", False)
        ),
        require_asset_mesh=bool(
            render_config.get("require_asset_mesh", True)
        ),
    )
    jobs: list[tuple[str, dict[str, Any]]] = []
    if scope in {"all", "sources"}:
        jobs.extend(("source", item) for item in dataset["sources"])
    if scope in {"all", "variants"}:
        jobs.extend(("variant", item) for item in dataset["variants"])
    start = max(0, int(start_index) - 1)
    jobs = jobs[start:]
    if limit is not None:
        jobs = jobs[: max(0, int(limit))]

    completed = 0
    resumed = 0
    failures: list[dict[str, Any]] = []
    for position, (kind, record) in enumerate(jobs, start=1):
        item_id = str(
            record["source_id"]
            if kind == "source"
            else record["variant_id"]
        )
        scene_path = Path(
            str(
                record["materialized_scene_path"]
                if kind == "source"
                else record["scene_path"]
            )
        )
        item_root = (
            output_root / "sources" / item_id
            if kind == "source"
            else output_root / "variants" / item_id
        )
        render_root = item_root / "render"
        provenance_path = render_root / "render_provenance.json"
        input_sha256 = file_sha256(scene_path)
        fingerprint = _render_fingerprint(
            input_sha256=input_sha256,
            render_config=render_config,
            blender_bin=blender_bin,
            asset_root=asset_root,
        )
        if resume and _render_ready(
            render_root,
            provenance_path=provenance_path,
            fingerprint=fingerprint,
        ):
            print(
                f"[{position:03d}/{len(jobs):03d}] {kind} {item_id}: resume",
                flush=True,
            )
            _update_record_from_render(record, render_root)
            resumed += 1
            continue
        print(
            f"[{position:03d}/{len(jobs):03d}] {kind} {item_id}: "
            f"{record['object_count']} objects",
            flush=True,
        )
        started = time.monotonic()
        try:
            manifest = renderer.render_scene(
                scene_path=scene_path,
                out_dir=render_root,
                asset_root=asset_root,
            )
            _validate_render_manifest(
                manifest,
                expected_object_count=int(record["object_count"]),
            )
            provenance = {
                "schema_version": "l3_mutation_render_provenance_v1",
                "item_kind": kind,
                "item_id": item_id,
                "status": "complete",
                "scene_sha256": input_sha256,
                "render_fingerprint": fingerprint,
                "elapsed_seconds": time.monotonic() - started,
                "render_config": deepcopy(render_config),
                "blender_bin": str(blender_bin),
                "asset_root": str(asset_root),
                "asset_coverage": manifest.get("asset_coverage"),
                "render_validation": manifest.get("render_validation"),
                "view_sha256": {
                    str(view["name"]): file_sha256(Path(str(view["path"])))
                    for view in manifest["views"]
                },
            }
            _write_json(provenance_path, provenance)
            (render_root / "failure.json").unlink(missing_ok=True)
            _update_record_from_render(record, render_root)
            if kind == "variant":
                _update_mutation_render_audit(
                    Path(str(record["mutation_manifest_path"])),
                    provenance=provenance,
                )
            completed += 1
        except Exception as exc:
            failure = {
                "schema_version": "l3_mutation_render_failure_v1",
                "item_kind": kind,
                "item_id": item_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "elapsed_seconds": time.monotonic() - started,
            }
            failures.append(failure)
            _write_json(render_root / "failure.json", failure)
            record["render_status"] = "failed"
            print(
                f"[{position:03d}/{len(jobs):03d}] {kind} {item_id}: "
                f"FAILED {failure['error_type']} · {failure['error']}",
                flush=True,
            )
            if kind == "variant":
                _update_mutation_render_failure(
                    Path(str(record["mutation_manifest_path"])),
                    failure=failure,
                )
            if not continue_on_error:
                _write_json(dataset_path, dataset)
                raise
        _write_json(dataset_path, dataset)

    all_source_complete = all(
        item.get("render_status") == "complete"
        for item in dataset["sources"]
    )
    all_variant_complete = all(
        item.get("render_status") == "complete"
        for item in dataset["variants"]
    )
    dataset["render_status"] = (
        "complete"
        if all_source_complete and all_variant_complete
        else "partial"
    )
    dataset["render_summary"] = {
        "source_complete": sum(
            item.get("render_status") == "complete"
            for item in dataset["sources"]
        ),
        "source_total": len(dataset["sources"]),
        "variant_complete": sum(
            item.get("render_status") == "complete"
            for item in dataset["variants"]
        ),
        "variant_total": len(dataset["variants"]),
        "failures": failures,
    }
    _write_json(dataset_path, dataset)
    return {
        "status": dataset["render_status"],
        "jobs_requested": len(jobs),
        "completed_now": completed,
        "resume_hits": resumed,
        "failure_count": len(failures),
        "failures": failures,
        **dataset["render_summary"],
    }


def _validate_render_manifest(
    manifest: dict[str, Any],
    *,
    expected_object_count: int,
) -> None:
    views = manifest.get("views")
    if not isinstance(views, list):
        raise ValueError("render manifest has no views")
    names = {str(item.get("name")) for item in views}
    required = {"top", "perspective", "identity_map"}
    if not required.issubset(names):
        raise ValueError(
            f"render manifest is missing views {sorted(required - names)}"
        )
    validation = manifest.get("render_validation")
    if not isinstance(validation, dict):
        raise ValueError("render manifest has no validation")
    if validation.get("blank_views"):
        raise ValueError(
            f"render contains blank views {validation['blank_views']}"
        )
    coverage = manifest.get("asset_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("render manifest has no asset coverage")
    if int(coverage.get("object_count") or -1) != expected_object_count:
        raise ValueError("render object count does not match scene")
    if int(coverage.get("asset_mesh_count") or -1) != expected_object_count:
        raise ValueError(
            "render used one or more bbox proxies instead of asset meshes"
        )
    identity = validation.get("identity_map")
    if (
        not isinstance(identity, dict)
        or identity.get("status") not in {"valid", "verified"}
    ):
        raise ValueError("identity-map validation did not pass")


def _render_ready(
    render_root: Path,
    *,
    provenance_path: Path,
    fingerprint: str,
) -> bool:
    if not provenance_path.is_file():
        return False
    provenance = _read_json(provenance_path)
    if (
        provenance.get("status") != "complete"
        or provenance.get("render_fingerprint") != fingerprint
    ):
        return False
    manifest_path = render_root / "render_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    try:
        expected = int(
            manifest.get("asset_coverage", {}).get("object_count")
        )
        _validate_render_manifest(
            manifest,
            expected_object_count=expected,
        )
    except (TypeError, ValueError):
        return False
    return (render_root / "scene.blend").is_file()


def _update_record_from_render(
    record: dict[str, Any],
    render_root: Path,
) -> None:
    manifest_path = render_root / "render_manifest.json"
    manifest = _read_json(manifest_path)
    record.update(
        {
            "render_status": "complete",
            "render_dir": str(render_root.resolve()),
            "render_manifest_path": str(manifest_path.resolve()),
            "blend_file": str(
                Path(str(manifest["blend_file"])).resolve()
            ),
            "view_paths": {
                str(view["name"]): str(Path(str(view["path"])).resolve())
                for view in manifest["views"]
            },
        }
    )


def _update_mutation_render_audit(
    path: Path,
    *,
    provenance: dict[str, Any],
) -> None:
    mutation = _read_json(path)
    mutation["render_validation"] = {
        "status": "complete",
        "asset_coverage": provenance["asset_coverage"],
        "render_validation": provenance["render_validation"],
        "view_sha256": provenance["view_sha256"],
        "elapsed_seconds": provenance["elapsed_seconds"],
    }
    _write_json(path, mutation)


def _update_mutation_render_failure(
    path: Path,
    *,
    failure: dict[str, Any],
) -> None:
    mutation = _read_json(path)
    mutation["render_validation"] = {
        "status": "failed",
        "failure": failure,
    }
    _write_json(path, mutation)


def _render_fingerprint(
    *,
    input_sha256: str,
    render_config: dict[str, Any],
    blender_bin: Path,
    asset_root: Path,
) -> str:
    import hashlib

    payload = {
        "scene_sha256": input_sha256,
        "render_config": render_config,
        "blender_bin": str(blender_bin),
        "blender_mtime_ns": blender_bin.stat().st_mtime_ns,
        "asset_root": str(asset_root),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _path(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


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
