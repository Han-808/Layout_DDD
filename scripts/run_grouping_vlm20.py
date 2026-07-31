#!/usr/bin/env python3
"""Run the active VLM grouping contract on 20 frozen rendered scenes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.grouping import (  # noqa: E402
    VLM_GROUPING_POLICY_ID,
    VLM_GROUPING_PROMPT_VERSION,
    group_scene,
    normalize_grouping_scene,
)
from benchmark.models import OpenAICompatibleModel  # noqa: E402
from scripts.grouping_blind30_contracts import (  # noqa: E402
    atomic_write_json,
    evidence_packet,
    file_sha256,
    json_sha256,
    read_json,
    repo_path,
)
from scripts.grouping_blind30_visuals import (  # noqa: E402
    draw_grouping_overlay,
)


SCHEMA_VERSION = "grouping_vlm20_experiment_v1"
RESULT_SCHEMA_VERSION = "grouping_vlm20_result_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiments"
        / "grouping_vlm20_visual_scope_v2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze the 20-case selection and build no model results.",
    )
    args = parser.parse_args()

    config = _load_config(args.config, output_override=args.output_root)
    output_root = Path(config["_output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = _prepare_dataset(config, resume=args.resume)
    if args.prepare_only:
        _print_summary(config, dataset, failures=[])
        return

    model = _build_model(config["model"])
    grouping_config = _load_yaml_object(
        Path(config["_grouping_config"])
    )
    failures: list[dict[str, Any]] = []
    for position, case in enumerate(dataset["cases"], start=1):
        case_id = str(case["case_id"])
        result_root = output_root / "cases" / case_id
        result_path = result_root / "result.json"
        expected_fingerprint = _input_fingerprint(
            case=case,
            grouping_config=grouping_config,
            model_config=config["model"],
        )
        if args.resume and _result_ready(
            result_path,
            expected_fingerprint=expected_fingerprint,
        ):
            print(
                f"[group {position:02d}/20] {case_id}: resume hit",
                flush=True,
            )
            continue

        print(
            f"[group {position:02d}/20] {case_id}: "
            f"{case['scene_type']} · {case['object_count']} objects",
            flush=True,
        )
        started = time.monotonic()
        try:
            source_case_root = Path(case["source_case_root"])
            input_manifest = read_json(
                source_case_root / "input" / "input_manifest.json"
            )
            scene = read_json(
                Path(input_manifest["materialized_scene_path"])
            )
            render_manifest = read_json(
                source_case_root / "render" / "render_manifest.json"
            )
            evidence = evidence_packet(
                input_manifest=input_manifest,
                render_manifest=render_manifest,
            )
            result = group_scene(
                scene,
                case={
                    "case_id": case_id,
                    "scene_id": case["source_scene_id"],
                    "scene_type": case["scene_type"],
                },
                visual_evidence=evidence,
                config=grouping_config,
                context={
                    "grouping_goal": (
                        "Create a complete downstream visual-evidence-scope "
                        "partition for per-group camera acquisition and "
                        "group-local metric evaluation."
                    ),
                    "identity_overlay_legend": {
                        alias: object_id
                        for object_id, alias in input_manifest[
                            "object_aliases"
                        ].items()
                    },
                },
                model=model,
            )
            normalized = normalize_grouping_scene(scene)
            result_root.mkdir(parents=True, exist_ok=True)
            preview = draw_grouping_overlay(
                normalized=normalized,
                aliases=input_manifest["object_aliases"],
                result=result,
                blind_label="updated VLM",
                output_path=result_root / "grouping_overlay.png",
            )
            record = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "experiment_id": config["experiment_id"],
                "case_id": case_id,
                "status": "complete",
                "backend": result.backend,
                "policy_id": result.policy_id,
                "prompt_version": result.provenance.get("prompt_version"),
                "input_fingerprint": expected_fingerprint,
                "elapsed_seconds": time.monotonic() - started,
                "visual_evidence": deepcopy(evidence),
                "result": result.to_dict(),
                "preview": preview,
                "audit": {
                    "model": config["model"]["model"],
                    "endpoint": config["model"]["endpoint"],
                    "image_count": len(evidence),
                    "scene_access": "read_only",
                },
            }
            atomic_write_json(result_path, record)
            (result_root / "failure.json").unlink(missing_ok=True)
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            }
            failures.append(failure)
            result_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(result_root / "failure.json", failure)
            print(
                f"[group {position:02d}/20] {case_id}: "
                f"FAILED {failure['error_type']}",
                flush=True,
            )
            if not args.continue_on_error:
                raise

    _build_gallery(config, dataset)
    _print_summary(config, dataset, failures=failures)
    if failures:
        raise SystemExit(1)


def _load_config(
    path: Path,
    *,
    output_override: Path | None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    config = _load_yaml_object(resolved)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    sample = config.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("sample must be an object")
    if int(sample.get("size", 0)) != 20:
        raise ValueError("sample.size must be exactly 20")
    if int(sample.get("per_stratum", 0)) != 4:
        raise ValueError("sample.per_stratum must be exactly 4")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    output_root = (
        output_override.expanduser().resolve()
        if output_override is not None
        else repo_path(PROJECT_ROOT, str(config["output_root"]))
    )
    config["_config_path"] = str(resolved)
    config["_output_root"] = str(output_root)
    config["_source_root"] = str(
        repo_path(PROJECT_ROOT, str(config["source_experiment_root"]))
    )
    config["_grouping_config"] = str(
        repo_path(PROJECT_ROOT, str(config["grouping_config"]))
    )
    return config


def _prepare_dataset(
    config: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass --resume to reuse it"
            )
        manifest = read_json(manifest_path)
        if (
            manifest.get("experiment_id") != config["experiment_id"]
            or manifest.get("sample_size") != 20
            or len(manifest.get("cases", [])) != 20
        ):
            raise ValueError("existing 20-case dataset manifest is invalid")
        return manifest

    source_root = Path(config["_source_root"])
    source_manifest = read_json(source_root / "dataset_manifest.json")
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for case in source_manifest["cases"]:
        by_stratum.setdefault(str(case["stratum"]), []).append(case)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        pool = list(by_stratum[stratum])
        rng = random.Random(
            json_sha256(
                {
                    "seed": int(config["seed"]),
                    "stratum": stratum,
                    "purpose": "vlm20_subset",
                }
            )
        )
        rng.shuffle(pool)
        if len(pool) < 4:
            raise ValueError(f"stratum {stratum!r} has fewer than 4 cases")
        selected.extend(pool[:4])
    order_rng = random.Random(
        json_sha256(
            {
                "seed": int(config["seed"]),
                "purpose": "vlm20_case_order",
            }
        )
    )
    order_rng.shuffle(selected)
    cases: list[dict[str, Any]] = []
    for index, source_case in enumerate(selected, start=1):
        source_case_id = str(source_case["case_id"])
        source_case_root = source_root / "cases" / source_case_id
        for required in (
            source_case_root / "input" / "input_manifest.json",
            source_case_root / "render" / "render_manifest.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        cases.append(
            {
                **deepcopy(source_case),
                "case_id": f"case_{index:03d}",
                "source_case_id": source_case_id,
                "source_case_root": str(source_case_root.resolve()),
            }
        )
    manifest = {
        "schema_version": "grouping_vlm20_dataset_v1",
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "sample_size": len(cases),
        "sampling_policy": (
            "four_seeded_cases_per_original_object_count_stratum"
        ),
        "source_dataset_fingerprint": source_manifest[
            "dataset_fingerprint"
        ],
        "cases": cases,
    }
    manifest["dataset_fingerprint"] = json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        output_root / "experiment_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "config_path": config["_config_path"],
            "config_sha256": file_sha256(Path(config["_config_path"])),
            "effective": {
                key: deepcopy(value)
                for key, value in config.items()
                if not key.startswith("_")
            },
            "resolved": {
                "source_experiment_root": config["_source_root"],
                "grouping_config": config["_grouping_config"],
                "output_root": config["_output_root"],
                "policy_id": VLM_GROUPING_POLICY_ID,
                "prompt_version": VLM_GROUPING_PROMPT_VERSION,
            },
        },
    )
    return manifest


def _build_model(config: dict[str, Any]) -> OpenAICompatibleModel:
    key_env = str(config["api_key_env"])
    if not os.environ.get(key_env):
        raise RuntimeError(
            f"required API credential is not available in this process: "
            f"{key_env}"
        )
    return OpenAICompatibleModel(
        name=str(config["name"]),
        endpoint=str(config["endpoint"]),
        model_id=str(config["model"]),
        api_key_env=key_env,
        max_tokens=int(config["max_tokens"]),
        context_length=int(config["context_length"]),
        timeout_seconds=int(config["timeout_seconds"]),
        response_format_json=bool(config["response_format_json"]),
        max_retries=int(config["max_retries"]),
        retry_backoff_seconds=float(config["retry_backoff_seconds"]),
        max_tokens_field=str(config["max_tokens_field"]),
        send_temperature=bool(config["send_temperature"]),
        require_api_key=True,
    )


def _input_fingerprint(
    *,
    case: dict[str, Any],
    grouping_config: dict[str, Any],
    model_config: dict[str, Any],
) -> str:
    source_case_root = Path(case["source_case_root"])
    input_manifest = read_json(
        source_case_root / "input" / "input_manifest.json"
    )
    render_manifest = read_json(
        source_case_root / "render" / "render_manifest.json"
    )
    evidence = evidence_packet(
        input_manifest=input_manifest,
        render_manifest=render_manifest,
    )
    return json_sha256(
        {
            "source_input_fingerprint": input_manifest["input_fingerprint"],
            "evidence": [
                {
                    "role": item["role"],
                    "sha256": file_sha256(Path(item["path"])),
                }
                for item in evidence
            ],
            "grouping_config": grouping_config,
            "policy_id": VLM_GROUPING_POLICY_ID,
            "prompt_version": VLM_GROUPING_PROMPT_VERSION,
            "model": {
                key: model_config[key]
                for key in (
                    "endpoint",
                    "model",
                    "max_tokens",
                    "context_length",
                    "response_format_json",
                )
            },
        }
    )


def _result_ready(
    path: Path,
    *,
    expected_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == RESULT_SCHEMA_VERSION
        and result.get("status") == "complete"
        and result.get("policy_id") == VLM_GROUPING_POLICY_ID
        and result.get("prompt_version") == VLM_GROUPING_PROMPT_VERSION
        and result.get("input_fingerprint") == expected_fingerprint
    )


def _build_gallery(
    config: dict[str, Any],
    dataset: dict[str, Any],
) -> None:
    output_root = Path(config["_output_root"])
    gallery_root = output_root / "review"
    assets_root = gallery_root / "assets"
    cases: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        case_id = str(case["case_id"])
        source_case_root = Path(case["source_case_root"])
        input_manifest = read_json(
            source_case_root / "input" / "input_manifest.json"
        )
        render_manifest = read_json(
            source_case_root / "render" / "render_manifest.json"
        )
        result = read_json(output_root / "cases" / case_id / "result.json")
        view_paths = {
            str(item.get("name")): Path(str(item.get("path")))
            for item in render_manifest.get("views", [])
            if isinstance(item, dict)
        }
        sources = {
            "perspective": view_paths["perspective"],
            "top": view_paths["top"],
            "identity": Path(input_manifest["identity_map_path"]),
            "grouping": output_root
            / "cases"
            / case_id
            / "grouping_overlay.png",
        }
        public_assets = assets_root / case_id
        public_assets.mkdir(parents=True, exist_ok=True)
        images: dict[str, str] = {}
        for name, source in sources.items():
            destination = public_assets / f"{name}.png"
            shutil.copy2(source, destination)
            images[name] = f"assets/{case_id}/{destination.name}"
        cases.append(
            {
                "case_id": case_id,
                "source_case_id": case["source_case_id"],
                "scene_type": case["scene_type"],
                "object_count": int(case["object_count"]),
                "stratum": case["stratum"],
                "images": images,
                "group_count": result["preview"]["group_count"],
                "groups": result["preview"]["groups"],
            }
        )
    atomic_write_json(
        gallery_root / "review_data.json",
        {
            "schema_version": "grouping_vlm20_gallery_v1",
            "experiment_id": config["experiment_id"],
            "policy_id": VLM_GROUPING_POLICY_ID,
            "prompt_version": VLM_GROUPING_PROMPT_VERSION,
            "scene_count": len(cases),
            "cases": cases,
        },
    )
    gallery_root.mkdir(parents=True, exist_ok=True)
    (gallery_root / "index.html").write_text(
        _gallery_html(config, cases),
        encoding="utf-8",
    )


def _gallery_html(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    cards = []
    for case in cases:
        image_cards = "".join(
            (
                f'<figure><img src="{html.escape(case["images"][name])}" '
                f'alt="{html.escape(title)}"><figcaption>{title}</figcaption>'
                "</figure>"
            )
            for name, title in (
                ("perspective", "Original · perspective"),
                ("top", "Original · top"),
                ("identity", "Object identity map"),
                ("grouping", "Updated VLM grouping"),
            )
        )
        groups = "".join(
            (
                '<div class="group">'
                f'<b style="color:{html.escape(group["color"])}">'
                f'{html.escape(group["display_group_id"])}</b> '
                + " · ".join(
                    html.escape(
                        f'{member["object_alias"]} '
                        f'({member["description"]})'
                    )
                    for member in group["members"]
                )
                + "</div>"
            )
            for group in case["groups"]
        )
        cards.append(
            '<section class="case">'
            f'<h2>{html.escape(case["case_id"])} · '
            f'{html.escape(case["scene_type"])} '
            f'<span>{case["object_count"]} objects · '
            f'{case["group_count"]} groups</span></h2>'
            f'<div class="images">{image_cards}</div>'
            f'<details><summary>Exact group membership</summary>{groups}'
            "</details></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Updated VLM grouping · 20 scenes</title>
<style>
:root{{--bg:#101419;--panel:#1a2027;--line:#34404c;--text:#edf2f7;--muted:#9bacbd;--blue:#62a9ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:14px 20px;background:#11171ded;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}}
h1{{margin:0;font-size:24px}}header p{{margin:4px 0 0;color:var(--muted)}}main{{max-width:1800px;margin:auto;padding:18px}}
.case{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin:0 0 18px}}
h2{{margin:0 0 12px;font-size:19px}}h2 span{{float:right;color:var(--muted);font-size:14px;font-weight:500}}
.images{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}figure{{margin:0;background:#11161c;border:1px solid var(--line);border-radius:8px;overflow:hidden}}
img{{display:block;width:100%;aspect-ratio:1.25;object-fit:contain;background:#222831;cursor:zoom-in}}figcaption{{padding:7px 9px;color:#bfd9f5}}
details{{margin-top:10px;border-top:1px solid var(--line);padding-top:9px}}summary{{cursor:pointer;color:var(--blue)}}.group{{padding:6px 0;border-bottom:1px solid #29323c}}
@media(max-width:1100px){{.images{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.images{{grid-template-columns:1fr}}h2 span{{float:none;display:block}}}}
</style></head><body><header><h1>Updated VLM grouping · 20 scenes</h1>
<p>{html.escape(VLM_GROUPING_POLICY_ID)} · {html.escape(VLM_GROUPING_PROMPT_VERSION)} · {html.escape(str(config["model"]["model"]))}</p></header>
<main>{''.join(cards)}</main>
<script>document.querySelectorAll('img').forEach(x=>x.onclick=()=>window.open(x.src,'_blank'));</script>
</body></html>
"""


def _print_summary(
    config: dict[str, Any],
    dataset: dict[str, Any],
    *,
    failures: list[dict[str, Any]],
) -> None:
    output_root = Path(config["_output_root"])
    complete = sum(
        (
            output_root / "cases" / str(case["case_id"]) / "result.json"
        ).is_file()
        for case in dataset["cases"]
    )
    summary = {
        "experiment_id": config["experiment_id"],
        "scene_count": len(dataset["cases"]),
        "complete": complete,
        "failed": len(failures),
        "policy_id": VLM_GROUPING_POLICY_ID,
        "prompt_version": VLM_GROUPING_PROMPT_VERSION,
        "model": config["model"]["model"],
        "output_root": str(output_root),
        "review_index": str(output_root / "review" / "index.html"),
        "failures": failures,
    }
    atomic_write_json(output_root / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be an object: {path}")
    return value


if __name__ == "__main__":
    main()
