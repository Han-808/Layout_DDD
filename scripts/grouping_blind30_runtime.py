#!/usr/bin/env python3
"""Rendering and three-backend execution for blind grouping."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from benchmark.grouping import group_scene, normalize_grouping_scene
from benchmark.models import OpenAICompatibleModel
from benchmark.rendering import BlenderRenderer
from scripts.grouping_blind30_contracts import (
    RESULT_SCHEMA_VERSION,
    ExperimentPaths,
    atomic_write_json,
    blind_label_for_backend,
    evidence_packet,
    file_sha256,
    grouping_input_fingerprint,
    json_sha256,
    load_backend_config,
    read_json,
    repo_path,
    required_object,
    sanitized_error,
)
from scripts.grouping_blind30_visuals import draw_grouping_overlay


def render_all(
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset: dict[str, Any],
    *,
    resume: bool,
    continue_on_error: bool,
    blender_override: Path | None,
    asset_override: Path | None,
) -> list[dict[str, Any]]:
    render_config = required_object(config.get("render"), "render")
    blender_bin = (
        blender_override.expanduser().resolve()
        if blender_override is not None
        else repo_path(
            paths.repo_root,
            str(render_config["blender_bin"]),
        )
    )
    asset_root = (
        asset_override.expanduser().resolve()
        if asset_override is not None
        else repo_path(
            paths.repo_root,
            str(render_config["asset_root"]),
        )
    )
    if not blender_bin.is_file() or not os.access(blender_bin, os.X_OK):
        raise FileNotFoundError(
            f"Blender executable is unavailable: {blender_bin}"
        )
    if not asset_root.is_dir():
        raise FileNotFoundError(
            f"asset root is unavailable: {asset_root}"
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
            render_config.get("require_asset_mesh", False)
        ),
    )
    failures: list[dict[str, Any]] = []
    for position, case in enumerate(dataset["cases"], start=1):
        case_id = str(case["case_id"])
        input_manifest = read_json(
            paths.case_root(case_id) / "input" / "input_manifest.json"
        )
        scene_path = Path(input_manifest["materialized_scene_path"])
        render_root = paths.case_root(case_id) / "render"
        provenance_path = render_root / "experiment_render.json"
        render_fingerprint = _render_fingerprint(
            input_manifest=input_manifest,
            render_config=render_config,
            blender_bin=blender_bin,
            asset_root=asset_root,
        )
        if resume and _render_ready(
            render_root,
            provenance_path=provenance_path,
            expected_fingerprint=render_fingerprint,
        ):
            print(
                f"[render {position:02d}/30] {case_id}: resume hit",
                flush=True,
            )
            continue
        print(
            f"[render {position:02d}/30] {case_id}: "
            f"{case['scene_type']} · {case['object_count']} objects",
            flush=True,
        )
        started = time.monotonic()
        try:
            manifest = renderer.render_scene(
                scene_path=scene_path,
                out_dir=render_root,
                asset_root=asset_root,
            )
            atomic_write_json(
                provenance_path,
                {
                    "case_id": case_id,
                    "status": "complete",
                    "render_fingerprint": render_fingerprint,
                    "scene_access": "read_only",
                    "render_config": deepcopy(render_config),
                    "blender_bin": str(blender_bin),
                    "asset_root": str(asset_root),
                    "elapsed_seconds": time.monotonic() - started,
                    "asset_coverage": manifest.get("asset_coverage"),
                    "render_validation": manifest.get(
                        "render_validation"
                    ),
                },
            )
            (render_root / "failure.json").unlink(missing_ok=True)
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "stage": "render",
                "elapsed_seconds": time.monotonic() - started,
                **sanitized_error(exc),
            }
            failures.append(failure)
            atomic_write_json(render_root / "failure.json", failure)
            print(
                f"[render {position:02d}/30] {case_id}: FAILED "
                f"{failure['error_type']}",
                flush=True,
            )
            if not continue_on_error:
                raise
    return failures


def run_grouping_backends(
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset: dict[str, Any],
    *,
    resume: bool,
    continue_on_error: bool,
    endpoint_override: str | None,
    model_override: str | None,
    api_key_env_override: str | None,
) -> list[dict[str, Any]]:
    method_key = read_json(paths.method_key)
    backend_configs = {
        backend: load_backend_config(
            config,
            backend=backend,
            repo_root=paths.repo_root,
        )
        for backend in config["backends"]
    }
    model_config = _effective_model_config(
        config,
        endpoint_override=endpoint_override,
        model_override=model_override,
        api_key_env_override=api_key_env_override,
    )
    model = _build_model(model_config)
    failures: list[dict[str, Any]] = []
    for case_position, case in enumerate(dataset["cases"], start=1):
        case_id = str(case["case_id"])
        case_root = paths.case_root(case_id)
        try:
            input_manifest = read_json(
                case_root / "input" / "input_manifest.json"
            )
            scene = read_json(
                Path(input_manifest["materialized_scene_path"])
            )
            render_manifest = read_json(
                case_root / "render" / "render_manifest.json"
            )
            evidence = evidence_packet(
                input_manifest=input_manifest,
                render_manifest=render_manifest,
            )
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "stage": "group_input",
                **sanitized_error(exc),
            }
            failures.append(failure)
            atomic_write_json(
                case_root / "grouping" / "input_failure.json",
                failure,
            )
            if not continue_on_error:
                raise
            continue
        evidence_paths = [Path(item["path"]) for item in evidence]
        normalized = normalize_grouping_scene(scene)
        for backend_position, backend in enumerate(
            config["backends"],
            start=1,
        ):
            blind_label = blind_label_for_backend(
                method_key,
                case_id=case_id,
                backend=backend,
            )
            backend_config = backend_configs[backend]
            fingerprint = grouping_input_fingerprint(
                input_manifest=input_manifest,
                evidence_paths=evidence_paths,
                backend=backend,
                backend_config=backend_config,
                model_config=model_config if backend == "vlm" else None,
            )
            result_root = case_root / "grouping" / backend
            result_path = result_root / "result.json"
            overlay_path = result_root / "overlay.png"
            if resume and _grouping_ready(
                result_path,
                expected_fingerprint=fingerprint,
            ):
                if not overlay_path.is_file():
                    stored = read_json(result_path)
                    draw_grouping_overlay(
                        normalized=normalized,
                        aliases=input_manifest["object_aliases"],
                        result=stored["result"],
                        blind_label=blind_label,
                        output_path=overlay_path,
                    )
                print(
                    f"[group {case_position:02d}/30 "
                    f"{backend_position}/3] {case_id}: resume hit",
                    flush=True,
                )
                continue
            print(
                f"[group {case_position:02d}/30 "
                f"{backend_position}/3] {case_id}: running",
                flush=True,
            )
            started = time.monotonic()
            try:
                result = group_scene(
                    scene,
                    case={
                        "case_id": case_id,
                        "scene_id": case["source_scene_id"],
                        "scene_type": case["scene_type"],
                    },
                    visual_evidence=evidence,
                    config=backend_config,
                    context={
                        "grouping_goal": (
                            "Create a complete evidence-scope partition for "
                            "later group-local metric evaluation."
                        ),
                        "identity_overlay_legend": {
                            alias: object_id
                            for object_id, alias in input_manifest[
                                "object_aliases"
                            ].items()
                        },
                    },
                    model=model if backend == "vlm" else None,
                )
                public_overlay = draw_grouping_overlay(
                    normalized=normalized,
                    aliases=input_manifest["object_aliases"],
                    result=result,
                    blind_label=blind_label,
                    output_path=overlay_path,
                )
                record = {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "experiment_id": config["_experiment_id"],
                    "case_id": case_id,
                    "source_scene_id": case["source_scene_id"],
                    "status": "complete",
                    "backend": backend,
                    "policy_id": result.policy_id,
                    "input_fingerprint": fingerprint,
                    "elapsed_seconds": time.monotonic() - started,
                    "visual_evidence": [
                        {
                            "role": item.get("role"),
                            "view_id": item.get("view_id"),
                            "path": item.get("path"),
                            "sha256": file_sha256(Path(item["path"])),
                        }
                        for item in evidence
                    ],
                    "result": result.to_dict(),
                    "blind_preview": public_overlay,
                    "audit": {
                        "model": (
                            model_config["model"]
                            if backend == "vlm"
                            else None
                        ),
                        "endpoint": (
                            model_config["endpoint"]
                            if backend == "vlm"
                            else None
                        ),
                        "api_key_env": (
                            model_config["api_key_env"]
                            if backend == "vlm"
                            else None
                        ),
                        "image_count": len(evidence),
                        "scene_access": "read_only",
                    },
                }
                atomic_write_json(result_path, record)
                (result_root / "failure.json").unlink(missing_ok=True)
            except Exception as exc:
                failure = {
                    "case_id": case_id,
                    "stage": "group",
                    "backend": backend,
                    "elapsed_seconds": time.monotonic() - started,
                    "input_fingerprint": fingerprint,
                    **sanitized_error(exc),
                }
                failures.append(failure)
                atomic_write_json(result_root / "failure.json", failure)
                print(
                    f"[group {case_position:02d}/30 "
                    f"{backend_position}/3] {case_id}: FAILED "
                    f"{failure['error_type']}",
                    flush=True,
                )
                if not continue_on_error:
                    raise
    return failures


def _build_model(config: dict[str, Any]) -> OpenAICompatibleModel:
    api_key_env = str(config["api_key_env"])
    if not os.environ.get(api_key_env):
        raise RuntimeError(
            f"required local proxy key environment variable is not set: "
            f"{api_key_env}"
        )
    return OpenAICompatibleModel(
        name=str(config["name"]),
        endpoint=str(config["endpoint"]),
        model_id=str(config["model"]),
        api_key_env=api_key_env,
        max_tokens=int(config["max_tokens"]),
        context_length=int(config["context_length"]),
        timeout_seconds=int(config["timeout_seconds"]),
        response_format_json=bool(config["response_format_json"]),
        max_retries=int(config["max_retries"]),
        retry_backoff_seconds=float(
            config["retry_backoff_seconds"]
        ),
        max_tokens_field=str(config["max_tokens_field"]),
        send_temperature=bool(config["send_temperature"]),
        require_api_key=True,
    )


def _effective_model_config(
    config: dict[str, Any],
    *,
    endpoint_override: str | None,
    model_override: str | None,
    api_key_env_override: str | None,
) -> dict[str, Any]:
    result = deepcopy(required_object(config.get("model"), "model"))
    if endpoint_override:
        result["endpoint"] = endpoint_override
    if model_override:
        result["model"] = model_override
    if api_key_env_override:
        result["api_key_env"] = api_key_env_override
    required = (
        "name",
        "endpoint",
        "model",
        "api_key_env",
        "max_tokens",
        "context_length",
        "timeout_seconds",
        "response_format_json",
        "max_retries",
        "retry_backoff_seconds",
        "max_tokens_field",
        "send_temperature",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"model config is missing fields {missing}")
    return result


def _render_fingerprint(
    *,
    input_manifest: dict[str, Any],
    render_config: dict[str, Any],
    blender_bin: Path,
    asset_root: Path,
) -> str:
    return json_sha256(
        {
            "input_fingerprint": input_manifest["input_fingerprint"],
            "render_config": render_config,
            "blender_bin": str(blender_bin),
            "asset_root": str(asset_root),
        }
    )


def _render_ready(
    render_root: Path,
    *,
    provenance_path: Path,
    expected_fingerprint: str,
) -> bool:
    manifest_path = render_root / "render_manifest.json"
    if not provenance_path.is_file() or not manifest_path.is_file():
        return False
    try:
        provenance = read_json(provenance_path)
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        provenance.get("status") != "complete"
        or provenance.get("render_fingerprint") != expected_fingerprint
    ):
        return False
    views = manifest.get("views")
    return bool(
        isinstance(views, list)
        and len(views) >= 2
        and all(
            isinstance(item, dict)
            and Path(str(item.get("path"))).is_file()
            for item in views
        )
    )


def _grouping_ready(
    result_path: Path,
    *,
    expected_fingerprint: str,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        record = read_json(result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        record.get("schema_version") == RESULT_SCHEMA_VERSION
        and record.get("status") == "complete"
        and record.get("input_fingerprint") == expected_fingerprint
        and isinstance(record.get("result"), dict)
    )
