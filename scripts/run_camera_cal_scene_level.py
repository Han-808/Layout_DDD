#!/usr/bin/env python3
"""Run the promptless L1/L3 camera-cal experiment at scene level.

The frozen camera-cal cases contain generation prompts, but this experiment
does not evaluate prompt fidelity and never supplies those prompts to the
Judge. L1 remains scene-level deterministic evidence plus conditional VLM
adjudication. L3 runs the existing metric-specific scope policy, judges every
eligible group required by that metric, and preserves the evaluator's existing
scene-level aggregation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.api.evaluation import run_evaluate  # noqa: E402
from benchmark.evaluator.generic_validity.mesh_geometry import (  # noqa: E402
    load_collision_geometry_manifest,
)
from benchmark.evaluator.profile import (  # noqa: E402
    DEFAULT_EVALUATION_PROFILE,
    L1,
    L2,
    L3,
    L4,
)
from benchmark.models import OpenAICompatibleModel  # noqa: E402
from benchmark.rendering import (  # noqa: E402
    CYCLES_DEVICES,
    RENDER_ENGINES,
    BlenderRenderer,
)
from benchmark.visual_judge import (  # noqa: E402
    CameraCandidatePreviewRenderer,
    CameraEvidenceProvider,
    CameraViewEvidenceRenderer,
    DeterministicLocalCameraSelector,
    build_openai_compatible_camera_selector,
    build_openai_compatible_vlm_judge,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.l3_prompts import (  # noqa: E402
    L3_METRIC_PROMPT_VERSION,
)
from benchmark.visual_judge.contracts import (  # noqa: E402
    response_schema_audit_from_exception,
)


RUNNER_SCHEMA_VERSION = "camera_cal_scene_level_runner_v2"
PLAN_SCHEMA_VERSION = "camera_cal_scene_level_plan_v1"
CASE_SCHEMA_VERSION = "camera_cal_scene_level_case_v1"
COMPARISON_SCHEMA_VERSION = "camera_cal_scene_comparison_v1"
SUMMARY_SCHEMA_VERSION = "camera_cal_scene_level_summary_v1"
L1_BINARY_FAILURE_POLICY = {
    "p0b_official_mode": False,
    "on_engineering_failure": "scene_unresolved_continue_l3_diagnostics",
    "binary_defects": "always_empty",
    "schema_repair_retry_count": 1,
}

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "camera_cal_scenesets"
)
DEFAULT_GROUPING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "grouping"
    / "vlm_visual_evidence_scope_v2.yaml"
)
DEFAULT_BLENDER_BIN = Path(
    os.environ.get(
        "BLENDER_BIN",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    )
)

L1_METRICS = ("collision", "oob", "support")
ANNOTATED_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
CANONICAL_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
EXPERIMENTAL_L3_METRICS = (
    "functional_consistency",
    "semantic_placement_consistency",
)

_CASE_ID_PATTERN = re.compile(r"N\d{3}")
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def main() -> None:
    args = parse_args()
    route = effective_model_route()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    grouping_config_path = args.grouping_config.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    metrics = normalize_metric_selection(args.metric)
    cases = discover_cases(
        dataset_root,
        case_ids=args.case_id,
        max_cases=args.max_cases,
    )
    if not grouping_config_path.is_file():
        raise FileNotFoundError(
            f"grouping config does not exist: {grouping_config_path}"
        )
    if not blender_bin.is_file():
        raise FileNotFoundError(f"Blender executable does not exist: {blender_bin}")

    renderer_config = renderer_config_from_args(args, blender_bin=blender_bin)
    control = resolved_control()
    experiment = build_experiment_plan(
        dataset_root=dataset_root,
        output_root=output_root,
        grouping_config_path=grouping_config_path,
        route=route,
        metrics=metrics,
        cases=cases,
        renderer_config=renderer_config,
        control=control.to_dict(),
        max_workers=args.max_workers,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "experiment_plan.json", experiment)

    started = time.monotonic()
    run_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "experiment_plan_sha256": json_sha256(experiment),
        "source_prompt_used": False,
        "layers_executed": [L1, L3],
        "layers_not_executed": [L2, L4],
        "cases": [],
    }
    atomic_write_json(output_root / "run_manifest.json", run_manifest)

    case_kwargs = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "grouping_config_path": grouping_config_path,
        "route": route,
        "metrics": metrics,
        "renderer_config": renderer_config,
        "control_config": control.to_dict(),
        "resume": args.resume,
    }
    if args.max_workers == 1:
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index:03d}/{len(cases):03d}] {case['case_id']} starting",
                flush=True,
            )
            try:
                record = run_case(case=case, **case_kwargs)
            except Exception as exc:
                failure = record_case_failure(
                    case=case,
                    output_root=output_root,
                    error=exc,
                )
                failures.append(failure)
                run_records.append(failure)
                print(
                    f"[{index:03d}/{len(cases):03d}] "
                    f"{case['case_id']} FAILED {failure['error_type']}",
                    flush=True,
                )
                if not args.continue_on_error:
                    break
            else:
                run_records.append(record)
                print(
                    f"[{index:03d}/{len(cases):03d}] "
                    f"{case['case_id']} {record['status']}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_case: dict[Future[dict[str, Any]], dict[str, Any]] = {
                executor.submit(run_case, case=case, **case_kwargs): case
                for case in cases
            }
            for completed, future in enumerate(
                as_completed(future_to_case),
                start=1,
            ):
                case = future_to_case[future]
                try:
                    record = future.result()
                except Exception as exc:
                    failure = record_case_failure(
                        case=case,
                        output_root=output_root,
                        error=exc,
                    )
                    failures.append(failure)
                    run_records.append(failure)
                    print(
                        f"[{completed:03d}/{len(cases):03d}] "
                        f"{case['case_id']} FAILED {failure['error_type']}",
                        flush=True,
                    )
                    if not args.continue_on_error:
                        for pending in future_to_case:
                            pending.cancel()
                        break
                else:
                    run_records.append(record)
                    print(
                        f"[{completed:03d}/{len(cases):03d}] "
                        f"{case['case_id']} {record['status']}",
                        flush=True,
                    )

    ordered_records = sorted(
        run_records,
        key=lambda item: str(item.get("case_id") or ""),
    )
    elapsed = time.monotonic() - started
    summary = build_summary(
        case_records=ordered_records,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )
    atomic_write_json(output_root / "summary.json", summary)
    run_manifest.update(
        status="failed" if failures else "complete",
        completed_at=utc_now(),
        elapsed_seconds=elapsed,
        cases=ordered_records,
        summary_path=str((output_root / "summary.json").resolve()),
    )
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(summary["totals"], indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or resumable run-level output directory.",
    )
    parser.add_argument(
        "--grouping-config",
        type=Path,
        default=DEFAULT_GROUPING_CONFIG,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Repeat to select cases. Omit to run every ready case.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=ANNOTATED_L3_METRICS,
        default=[],
        help="Repeat to select L3 metrics. Omit to run all five annotations.",
    )
    parser.add_argument("--max-cases", type=positive_int, default=None)
    parser.add_argument("--max-workers", type=positive_int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=DEFAULT_BLENDER_BIN,
    )
    parser.add_argument("--render-width", type=positive_int, default=768)
    parser.add_argument("--render-height", type=positive_int, default=768)
    parser.add_argument(
        "--render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument(
        "--cycles-device",
        choices=CYCLES_DEVICES,
        default="CPU",
    )
    parser.add_argument("--cycles-samples", type=positive_int, default=16)
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preview-render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--preview-width", type=positive_int, default=256)
    parser.add_argument("--preview-height", type=positive_int, default=256)
    parser.add_argument(
        "--preview-cycles-samples",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--blender-timeout-seconds",
        type=positive_int,
        default=900,
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def effective_model_route(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    required = ("JUDGE_ENDPOINT", "JUDGE_MODEL", "JUDGE_API_KEY_ENV")
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "explicit runtime model routing is required; missing "
            + ", ".join(missing)
        )
    endpoint = str(env["JUDGE_ENDPOINT"]).strip().rstrip("/")
    model = str(env["JUDGE_MODEL"]).strip()
    api_key_env = str(env["JUDGE_API_KEY_ENV"]).strip()
    if not _ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise ValueError("JUDGE_API_KEY_ENV must name a valid environment variable")
    if endpoint in {
        "http://127.0.0.1:4000",
        "http://127.0.0.1:4000/v1",
        "http://localhost:4000",
        "http://localhost:4000/v1",
    }:
        raise RuntimeError(
            "port 4000 is the stale LiteLLM route; set JUDGE_ENDPOINT "
            "explicitly to the intended endpoint"
        )
    if not str(env.get(api_key_env) or ""):
        raise RuntimeError(
            f"required API credential is not available in this process: "
            f"{api_key_env}"
        )
    return {
        "endpoint": endpoint,
        "model": model,
        "api_key_env": api_key_env,
        "authorization_configured": True,
    }


def normalize_metric_selection(values: Iterable[str]) -> tuple[str, ...]:
    selected = list(dict.fromkeys(str(value) for value in values))
    if not selected:
        return ANNOTATED_L3_METRICS
    unknown = sorted(set(selected) - set(ANNOTATED_L3_METRICS))
    if unknown:
        raise ValueError(f"unknown L3 metrics: {unknown}")
    return tuple(
        metric for metric in ANNOTATED_L3_METRICS if metric in selected
    )


def discover_cases(
    dataset_root: Path,
    *,
    case_ids: Iterable[str] = (),
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"camera-cal dataset does not exist: {root}")
    selected_ids = list(dict.fromkeys(str(value) for value in case_ids))
    invalid_ids = [
        case_id
        for case_id in selected_ids
        if not _CASE_ID_PATTERN.fullmatch(case_id)
    ]
    if invalid_ids:
        raise ValueError(f"invalid camera-cal case IDs: {invalid_ids}")

    discovered: dict[str, dict[str, Any]] = {}
    for case_root in sorted(root.iterdir()):
        if not case_root.is_dir() or not _CASE_ID_PATTERN.fullmatch(case_root.name):
            continue
        manifest_path = case_root / "case_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "ready":
            continue
        case_id = str(manifest.get("case_id") or case_root.name)
        required = case_paths(case_root, manifest)
        missing = [
            name
            for name, path in required.items()
            if name != "render_manifest" and not path.is_file()
        ]
        if missing:
            continue
        discovered[case_id] = {
            "case_id": case_id,
            "case_root": str(case_root),
            "scene_type": manifest.get("scene_type"),
            "object_count": manifest.get("object_count"),
            "semantic_content_fingerprint": manifest.get(
                "semantic_content_fingerprint"
            ),
        }
    if selected_ids:
        missing_ids = [case_id for case_id in selected_ids if case_id not in discovered]
        if missing_ids:
            raise ValueError(f"requested cases are not ready: {missing_ids}")
        cases = [discovered[case_id] for case_id in selected_ids]
    else:
        cases = [discovered[case_id] for case_id in sorted(discovered)]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("no ready camera-cal cases were selected")
    return cases


def case_paths(
    case_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    paths = manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    evidence = paths.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "scene": case_root
        / str(paths.get("canonical_scene") or "scene/canonical_scene.json"),
        "blend": case_root / str(paths.get("blend") or "prepared/evaluation.blend"),
        "annotation": case_root
        / str(paths.get("annotation") or "annotation.json"),
        "perspective": case_root
        / str(evidence.get("perspective") or "evidence/standardized_perspective.png"),
        "top": case_root
        / str(evidence.get("top") or "evidence/standardized_top.png"),
        "identity": case_root
        / str(evidence.get("identity") or "evidence/standardized_identity_map.png"),
        "render_manifest": case_root / "evidence" / "prepared_render_manifest.json",
        "collision_geometry": (
            case_root / "evidence" / "collision_geometry_manifest.json"
        ),
    }


def renderer_config_from_args(
    args: argparse.Namespace,
    *,
    blender_bin: Path,
) -> dict[str, Any]:
    return {
        "blender_bin": str(blender_bin),
        "timeout_seconds": int(args.blender_timeout_seconds),
        "width": int(args.render_width),
        "height": int(args.render_height),
        "render_engine": str(args.render_engine),
        "cycles_device": str(args.cycles_device),
        "cycles_samples": int(args.cycles_samples),
        "cycles_denoising": bool(args.cycles_denoising),
        "preview_render_engine": str(args.preview_render_engine),
        "preview_width": int(args.preview_width),
        "preview_height": int(args.preview_height),
        "preview_cycles_samples": int(args.preview_cycles_samples),
    }


def resolved_control() -> Any:
    return resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "deterministic_then_vlm",
                "deterministic": {
                    "max_rounds": 1,
                    "candidate_budget": 6,
                    "max_selected_views": 2,
                },
                "vlm": {
                    "max_rounds": 1,
                    "selection_mode": "repair_plan",
                    "max_selected_views": 2,
                },
                "total": {
                    "max_evidence_rounds": 2,
                    "max_total_images": 6,
                    "max_selector_calls": 3,
                    "max_camera_actions": 2,
                },
            },
            "budgets": {
                "max_evidence_rounds": 2,
                "max_views_per_round": 2,
                "max_total_images": 6,
                "max_selector_calls": 3,
                "max_camera_actions": 2,
            },
        },
        existing_max_views=2,
        existing_max_steps=1,
        existing_selector_available=True,
        judge_max_images=5,
    )


def build_experiment_plan(
    *,
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    cases: list[dict[str, Any]],
    renderer_config: dict[str, Any],
    control: dict[str, Any],
    max_workers: int,
    resume: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_cases_read_only": True,
        "source_prompt_used": False,
        "prompt_policy": "metric_rubrics_only_no_generation_prompt",
        "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "layers": {
            L1: {
                "enabled": True,
                "scope": "scene_level",
                "metrics": list(L1_METRICS),
                "backend": "deterministic_evidence_plus_conditional_vlm",
                "binary_failure_policy": deepcopy(
                    L1_BINARY_FAILURE_POLICY
                ),
            },
            L2: {
                "enabled": False,
                "reason": "promptless_camera_cal_experiment",
            },
            L3: {
                "enabled": True,
                "metrics": list(metrics),
                "scope": "metric_policy_then_scene_level_aggregation",
            },
            L4: {"enabled": False},
        },
        "model_route": safe_route_manifest(route),
        "grouping": {
            "config_path": str(grouping_config_path),
            "config_sha256": file_sha256(grouping_config_path),
        },
        "renderer": deepcopy(renderer_config),
        "control": deepcopy(control),
        "max_workers": max_workers,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "case_count": len(cases),
        "cases": deepcopy(cases),
    }


def run_case(
    *,
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    del dataset_root
    case_id = str(case["case_id"])
    source_root = Path(str(case["case_root"])).resolve()
    case_manifest = read_json(source_root / "case_manifest.json")
    paths = case_paths(source_root, case_manifest)
    annotation = read_json(paths["annotation"])
    scene = read_json(paths["scene"])
    grouping_config = read_yaml_object(grouping_config_path)
    fingerprint = case_input_fingerprint(
        case=case,
        case_manifest=case_manifest,
        paths=paths,
        route=route,
        metrics=metrics,
        grouping_config=grouping_config,
        renderer_config=renderer_config,
        control_config=control_config,
    )
    case_out = output_root / "cases" / case_id
    existing_manifest_path = case_out / "case_run_manifest.json"
    if existing_manifest_path.is_file():
        existing = read_json(existing_manifest_path)
        if resume and resumable_case(
            existing,
            expected_fingerprint=fingerprint,
            case_out=case_out,
        ):
            return {
                "case_id": case_id,
                "status": "resumed",
                "input_fingerprint": fingerprint,
                "elapsed_seconds": float(existing.get("elapsed_seconds") or 0.0),
                "grouping_status": existing.get("grouping_status"),
                "l1_status": existing.get("l1_status"),
                "l3_status": existing.get("l3_status"),
                "final_decision_status": existing.get(
                    "final_decision_status"
                ),
                "l1_engineering_failure": bool(
                    existing.get("l1_engineering_failure")
                ),
                "l1_engineering_failure_count": int(
                    existing.get("l1_engineering_failure_count") or 0
                ),
                "binary_response_schema_validation": deepcopy(
                    existing.get("binary_response_schema_validation") or {}
                ),
                "scene_comparison_path": str(
                    (case_out / "scene_comparison.json").resolve()
                ),
                "scene_quality_report_path": str(
                    (case_out / "scene_quality_report.json").resolve()
                ),
                "l1_report_path": str((case_out / "l1_report.json").resolve()),
                "control_manifest_path": str(
                    (case_out / "control_manifest.json").resolve()
                ),
            }
        if resume:
            raise RuntimeError(
                f"{case_id} existing output fingerprint does not match; "
                "use a new --output-root"
            )
        raise FileExistsError(
            f"{case_id} output already exists; use --resume or a new --output-root"
        )

    case_out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    case_run_manifest = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "input_fingerprint": fingerprint,
        "source_case_root": str(source_root),
        "source_case_read_only": True,
        "source_prompt_used": False,
        "model_route": safe_route_manifest(route),
        "selected_l3_metrics": list(metrics),
        "l1_binary_failure_policy": deepcopy(
            L1_BINARY_FAILURE_POLICY
        ),
    }
    atomic_write_json(existing_manifest_path, case_run_manifest)

    collision_geometry = load_collision_geometry_manifest(
        paths["collision_geometry"]
    )
    collision_geometry["manifest_path"] = str(
        paths["collision_geometry"].resolve()
    )
    identity_legend = identity_legend_from_manifest(paths["render_manifest"])
    grouping_evidence = grouping_evidence_packet(
        paths=paths,
        identity_legend=identity_legend,
    )
    overview_evidence = {
        "global": [
            str(paths["perspective"].resolve()),
            str(paths["top"].resolve()),
        ]
    }

    judge_config = model_config(route, role="judge")
    selector_config = model_config(route, role="camera-selector")
    grouping_model = build_grouping_model(route)
    raw_judge = build_openai_compatible_vlm_judge(judge_config)
    vlm_selector = build_openai_compatible_camera_selector(selector_config)
    renderer = BlenderRenderer(**renderer_config)
    l1_provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=paths["blend"],
        out_dir=case_out / "l1_camera",
        mode="auto",
        selector=None,
        max_views=2,
        max_steps=1,
        candidate_count=6,
        collision_overlay=True,
        collision_contour=True,
        collision_geometry=collision_geometry,
    )
    l3_provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=paths["blend"],
        out_dir=case_out / "l3_initial_camera",
        mode="visibility_ranked",
        selector=None,
        max_views=2,
        max_steps=1,
        candidate_count=6,
        collision_overlay=False,
        collision_contour=False,
        collision_geometry=collision_geometry,
        active_repair=False,
    )
    deterministic_selector = DeterministicLocalCameraSelector(
        candidate_policy=l3_provider.candidate_policy
    )
    evidence_renderer = CameraViewEvidenceRenderer(
        renderer=renderer,
        blend_file=paths["blend"],
        out_dir=case_out / "repair_camera",
    )
    preview_renderer = CameraCandidatePreviewRenderer(
        renderer=renderer,
        blend_file=paths["blend"],
        out_dir=case_out / "repair_camera",
    )

    report = run_evaluate(
        scene=scene,
        out=case_out / "evaluation_report.json",
        scene_request=promptless_scene_request(scene, case),
        collision_geometry=collision_geometry,
        render_evidence=overview_evidence,
        grouping_visual_evidence=grouping_evidence,
        grouping_identity_legend=identity_legend,
        vlm_judge=raw_judge,
        grouping_model=grouping_model,
        evaluation_profile=promptless_l1_l3_profile(),
        p0b_official_mode=L1_BINARY_FAILURE_POLICY["p0b_official_mode"],
        p0b_local_view_provider=l1_provider,
        l3_initial_evidence_provider=l3_provider,
        camera_selector=vlm_selector,
        deterministic_camera_selector=deterministic_selector,
        vlm_camera_selector=vlm_selector,
        evidence_renderer=evidence_renderer,
        candidate_preview_renderer=preview_renderer,
        scene_quality_config=scene_quality_config(metrics),
        asset_policy=camera_cal_asset_policy(),
        specification_contract=None,
        authorized_deviations=[],
        vlm_evaluation_control=control_config,
    )
    grouping_report = deepcopy(report["reports"]["object_grouping"])
    l1_report = deepcopy(report["layer_reports"][L1])
    l3_report = deepcopy(report["reports"]["scene_quality"])
    control_manifest = deepcopy(
        report["evaluation_config"]["vlm_evaluation_control"]
    )
    l1_failures = collect_l1_engineering_failures(l1_report)
    schema_validation = binary_schema_validation_summary(l1_report)
    final_decision_status = (
        "resolved"
        if l1_report.get("status") == "evaluated" and not l1_failures
        else "unresolved"
    )
    diagnostic_reason = (
        "l1_engineering_failure"
        if l1_failures
        else "l1_unresolved"
        if final_decision_status == "unresolved"
        else None
    )
    comparison = build_scene_comparison(
        case_id=case_id,
        annotation=annotation,
        scene_quality_report=l3_report,
        metrics=metrics,
    )
    comparison["diagnostic_only"] = (
        final_decision_status == "unresolved"
    )
    comparison["diagnostic_reason"] = diagnostic_reason
    l1_diagnostics = {
        "policy": deepcopy(L1_BINARY_FAILURE_POLICY),
        "final_decision_status": final_decision_status,
        "engineering_failure_count": len(l1_failures),
        "engineering_failures": l1_failures,
        "response_schema_validation": schema_validation,
        "l3_diagnostics_completed": True,
    }
    report["runner_outcome"] = {
        "final_decision_status": final_decision_status,
        "l1_engineering_failure": bool(l1_failures),
        "l3_results_are_diagnostic_only": (
            final_decision_status == "unresolved"
        ),
        "l1_diagnostics_path": str(
            (case_out / "l1_diagnostics.json").resolve()
        ),
    }
    elapsed = time.monotonic() - started

    atomic_write_json(case_out / "evaluation_report.json", report)
    atomic_write_json(case_out / "grouping.json", grouping_report)
    atomic_write_json(case_out / "l1_report.json", l1_report)
    atomic_write_json(case_out / "l1_diagnostics.json", l1_diagnostics)
    atomic_write_json(case_out / "scene_quality_report.json", l3_report)
    atomic_write_json(case_out / "scene_comparison.json", comparison)
    atomic_write_json(case_out / "control_manifest.json", control_manifest)
    case_run_manifest.update(
        status="complete",
        completed_at=utc_now(),
        elapsed_seconds=elapsed,
        grouping_status=grouping_report.get("status"),
        l1_status=l1_report.get("status"),
        l3_status=l3_report.get("status"),
        final_decision_status=final_decision_status,
        l1_engineering_failure=bool(l1_failures),
        l1_engineering_failure_count=len(l1_failures),
        binary_response_schema_validation=schema_validation,
        paths={
            "evaluation_report": str(
                (case_out / "evaluation_report.json").resolve()
            ),
            "grouping": str((case_out / "grouping.json").resolve()),
            "l1_report": str((case_out / "l1_report.json").resolve()),
            "l1_diagnostics": str(
                (case_out / "l1_diagnostics.json").resolve()
            ),
            "scene_quality_report": str(
                (case_out / "scene_quality_report.json").resolve()
            ),
            "scene_comparison": str(
                (case_out / "scene_comparison.json").resolve()
            ),
            "control_manifest": str(
                (case_out / "control_manifest.json").resolve()
            ),
        },
    )
    atomic_write_json(existing_manifest_path, case_run_manifest)
    return {
        "case_id": case_id,
        "status": "complete",
        "input_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "grouping_status": grouping_report.get("status"),
        "l1_status": l1_report.get("status"),
        "l3_status": l3_report.get("status"),
        "final_decision_status": final_decision_status,
        "l1_engineering_failure": bool(l1_failures),
        "l1_engineering_failure_count": len(l1_failures),
        "binary_response_schema_validation": schema_validation,
        "scene_comparison_path": str(
            (case_out / "scene_comparison.json").resolve()
        ),
        "scene_quality_report_path": str(
            (case_out / "scene_quality_report.json").resolve()
        ),
        "l1_report_path": str((case_out / "l1_report.json").resolve()),
        "control_manifest_path": str(
            (case_out / "control_manifest.json").resolve()
        ),
    }


def model_config(route: dict[str, Any], *, role: str) -> dict[str, Any]:
    max_images = 8 if role == "camera-selector" else 5
    return {
        "name": f"camera-cal-{role}",
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "temperature": 0.0,
        "send_temperature": False,
        "max_tokens": 2048,
        "timeout_seconds": 3000,
        "response_format_json": False,
        "max_retries": 1,
        "retry_backoff_seconds": 1.0,
        "max_images": max_images,
        "max_preview_images": max_images,
        "max_context_chars": 30000,
        "require_api_key": True,
    }


def build_grouping_model(route: dict[str, Any]) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        name="camera-cal-grouping",
        endpoint=str(route["endpoint"]),
        model_id=str(route["model"]),
        api_key_env=str(route["api_key_env"]),
        temperature=0.0,
        max_tokens=2048,
        timeout_seconds=3000,
        response_format_json=False,
        max_retries=1,
        retry_backoff_seconds=1.0,
        send_temperature=False,
        require_api_key=True,
    )


def promptless_scene_request(
    scene: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": scene.get("request_id"),
        "instruction": "",
        "scene_type": scene.get("scene_type") or case.get("scene_type"),
        "prompt_granularity": "fine_grained",
        "metadata": {
            "promptless_camera_cal": True,
            "generation_prompt_withheld_from_evaluator": True,
        },
    }


def promptless_l1_l3_profile() -> dict[str, Any]:
    profile = deepcopy(DEFAULT_EVALUATION_PROFILE)
    profile["layer_weights"] = {
        L1: 7.0 / 15.0,
        L2: 0.0,
        L3: 8.0 / 15.0,
        L4: 0.0,
    }
    profile[L2]["enabled"] = False
    for metric in profile[L2]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def scene_quality_config(metrics: tuple[str, ...]) -> dict[str, Any]:
    selected = set(metrics)
    return {
        "enabled": True,
        "metrics": {
            metric: {
                "enabled": metric in selected,
                "weight": 1.0 if metric in selected else 0.0,
            }
            for metric in ANNOTATED_L3_METRICS
        },
    }


def camera_cal_asset_policy() -> dict[str, Any]:
    return {
        "mode": "fixed_catalog_selection",
        "identity_owner": "benchmark",
        "category_selection_owner": "generator",
        "scale_owner": "generator",
        "appearance_owner": "generator",
        "arrangement_owner": "generator",
        "source": "camera_cal_experiment_protocol",
    }


def identity_legend_from_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    manifest = read_json(path)
    legend = manifest.get("identity_legend")
    if not isinstance(legend, dict):
        return {}
    return {
        str(alias): str(object_id)
        for alias, object_id in legend.items()
        if str(alias).strip() and str(object_id).strip()
    }


def grouping_evidence_packet(
    *,
    paths: dict[str, Path],
    identity_legend: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(paths["perspective"].resolve()),
            "role": "global_perspective_rgb",
            "representation": "rgb",
            "view_id": "global_perspective",
            "camera_scope": "global",
        },
        {
            "path": str(paths["top"].resolve()),
            "role": "global_top_rgb",
            "representation": "rgb",
            "view_id": "global_top",
            "camera_scope": "global",
        },
        {
            "path": str(paths["identity"].resolve()),
            "role": "global_identity_overlay",
            "representation": "identity_map",
            "view_id": "global_identity",
            "camera_scope": "global",
            "identity_overlay": True,
            "identity_legend": deepcopy(identity_legend),
        },
    ]


def case_input_fingerprint(
    *,
    case: dict[str, Any],
    case_manifest: dict[str, Any],
    paths: dict[str, Path],
    route: dict[str, Any],
    metrics: tuple[str, ...],
    grouping_config: dict[str, Any],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
) -> str:
    critical = case_manifest.get("critical_artifact_hashes")
    critical = critical if isinstance(critical, dict) else {}
    prompt_path = (
        PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "l3_prompts.py"
    )
    return json_sha256(
        {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "semantic_content_fingerprint": case.get(
                "semantic_content_fingerprint"
            ),
            "canonical_scene_sha256": file_sha256(paths["scene"]),
            "annotation_sha256": file_sha256(paths["annotation"]),
            "blend_sha256": critical.get("blend"),
            "evidence_sha256": {
                name: file_sha256(paths[name])
                for name in ("perspective", "top", "identity")
            },
            "collision_geometry_manifest_sha256": file_sha256(
                paths["collision_geometry"]
            ),
            "grouping_config": grouping_config,
            "model_route": safe_route_manifest(route),
            "selected_l3_metrics": list(metrics),
            "source_prompt_used": False,
            "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "l3_prompt_source_sha256": file_sha256(prompt_path),
            "profile": promptless_l1_l3_profile(),
            "scene_quality_config": scene_quality_config(metrics),
            "asset_policy": camera_cal_asset_policy(),
            "l1_binary_failure_policy": deepcopy(
                L1_BINARY_FAILURE_POLICY
            ),
            "control": control_config,
            "renderer": {
                key: value
                for key, value in renderer_config.items()
                if key != "blender_bin"
            },
        }
    )


def resumable_case(
    manifest: dict[str, Any],
    *,
    expected_fingerprint: str,
    case_out: Path,
) -> bool:
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("input_fingerprint") == expected_fingerprint
        and (case_out / "evaluation_report.json").is_file()
        and (case_out / "grouping.json").is_file()
        and (case_out / "l1_report.json").is_file()
        and (case_out / "l1_diagnostics.json").is_file()
        and (case_out / "scene_quality_report.json").is_file()
        and (case_out / "scene_comparison.json").is_file()
        and (case_out / "control_manifest.json").is_file()
    )


def build_scene_comparison(
    *,
    case_id: str,
    annotation: dict[str, Any],
    scene_quality_report: dict[str, Any],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    annotation_metrics = annotation.get("metrics")
    annotation_metrics = (
        annotation_metrics if isinstance(annotation_metrics, dict) else {}
    )
    report_metrics = scene_quality_report.get("metrics")
    report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
    comparisons: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        human = annotation_metrics.get(metric)
        human = human if isinstance(human, dict) else {}
        model = report_metrics.get(metric)
        model = model if isinstance(model, dict) else {}
        expected = (
            "invalid"
            if human.get("anomaly") is True
            else "valid"
            if human.get("anomaly") is False
            else "unresolved"
        )
        predicted = metric_prediction(model)
        unclear = human.get("unclear") is True
        evaluated = predicted in {"valid", "invalid"}
        included = bool(not unclear and expected in {"valid", "invalid"} and evaluated)
        comparisons[metric] = {
            "human": {
                "expected": expected,
                "anomaly": human.get("anomaly"),
                "unclear": unclear,
                "affected_object_ids": list(
                    human.get("affected_object_ids") or []
                ),
                "issue": human.get("issue"),
            },
            "model": {
                "prediction": predicted,
                "status": model.get("status"),
                "score": model.get("score"),
                "reason": model.get("reason"),
                "eligible_group_count": (
                    (model.get("coverage") or {}).get("eligible_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "resolved_group_count": (
                    (model.get("coverage") or {}).get("resolved_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "judge_call_count": model.get("judge_call_count"),
                "group_results": deepcopy(model.get("group_results") or []),
            },
            "included_in_accuracy": included,
            "matches": predicted == expected if included else None,
        }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "case_id": case_id,
        "source_prompt_used": False,
        "comparison_scope": "scene_level_metric_verdict",
        "metrics": comparisons,
    }


def metric_prediction(report: dict[str, Any]) -> str:
    if report.get("status") != "evaluated":
        return "unresolved"
    score = report.get("score")
    if score == 1.0:
        return "valid"
    if score == 0.0:
        return "invalid"
    return "unresolved"


def collect_l1_engineering_failures(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect fail-closed L1 transport/schema failures without hiding L3."""

    failures: list[dict[str, Any]] = []

    def visit(
        value: Any,
        *,
        path: tuple[str, ...],
        metric: str | None,
    ) -> None:
        if isinstance(value, dict):
            current_metric = (
                str(value["metric"])
                if value.get("metric") is not None
                else metric
            )
            if (
                value.get("route") == "vlm_adjudication_failed"
                or value.get("status") == "vlm_adjudication_failed"
            ):
                item = {
                    "path": ".".join(path),
                    "metric": current_metric,
                    "route": value.get("route"),
                    "status": value.get("status"),
                    "error": value.get("adjudication_error")
                    or (
                        value.get("evidence", {}).get("error")
                        if isinstance(value.get("evidence"), dict)
                        else None
                    ),
                }
                evidence = value.get("evidence")
                evidence = evidence if isinstance(evidence, dict) else {}
                audit = (
                    value.get("adjudication_failure_audit")
                    or evidence.get("adjudication_failure_audit")
                )
                if isinstance(audit, dict):
                    item["response_schema_validation"] = deepcopy(audit)
                failures.append(item)
            for key, child in value.items():
                visit(
                    child,
                    path=(*path, str(key)),
                    metric=current_metric,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=(*path, str(index)),
                    metric=metric,
                )

    visit(report, path=("l1",), metric=None)
    return failures


def binary_schema_validation_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    """Count binary response attempts separately from logical Judge calls."""

    audits: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in (
                "response_schema_validation",
                "adjudication_failure_audit",
            ):
                audit = value.get(key)
                if isinstance(audit, dict) and audit.get("policy") == (
                    "single_schema_repair_retry_v1"
                ):
                    audits.append(audit)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return {
        "logical_binary_judge_calls": len(audits),
        "response_attempts": sum(
            int(item.get("attempt_count") or 0)
            for item in audits
        ),
        "schema_repair_retries": sum(
            int(item.get("repair_retry_count") or 0)
            for item in audits
        ),
        "schema_repair_recoveries": sum(
            item.get("recovered") is True for item in audits
        ),
        "schema_repair_failures": sum(
            item.get("repair_retry_count") == 1
            and item.get("recovered") is False
            for item in audits
        ),
    }


def record_case_failure(
    *,
    case: dict[str, Any],
    output_root: Path,
    error: Exception,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_out = output_root / "cases" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": utc_now(),
    }
    schema_audit = response_schema_audit_from_exception(error)
    if schema_audit is not None:
        failure["response_schema_validation"] = schema_audit
    atomic_write_json(case_out / "failure.json", failure)
    manifest_path = case_out / "case_run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update(
            status="failed",
            completed_at=failure["failed_at"],
            final_decision_status="unresolved",
            error_type=failure["error_type"],
            error=failure["error"],
        )
        if schema_audit is not None:
            manifest["binary_response_schema_validation"] = schema_audit
        atomic_write_json(manifest_path, manifest)
    return failure


def build_summary(
    *,
    case_records: list[dict[str, Any]],
    metrics: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    metric_summaries = {
        metric: empty_metric_summary(total=len(case_records))
        for metric in metrics
    }
    successful = 0
    grouping_failures = 0
    final_unresolved = 0
    l1_engineering_failure_cases = 0
    binary_logical_judge_calls = 0
    binary_response_attempts = 0
    binary_schema_repair_retries = 0
    binary_schema_repair_recoveries = 0
    binary_schema_repair_failures = 0
    total_judge_calls = 0
    total_selector_calls = 0
    latencies: list[float] = []
    for record in case_records:
        if record.get("status") not in {"complete", "resumed"}:
            for summary in metric_summaries.values():
                summary["case_failures"] += 1
            continue
        successful += 1
        if record.get("final_decision_status") == "unresolved":
            final_unresolved += 1
        if record.get("l1_engineering_failure") is True:
            l1_engineering_failure_cases += 1
        binary_schema = record.get(
            "binary_response_schema_validation"
        )
        binary_schema = (
            binary_schema if isinstance(binary_schema, dict) else {}
        )
        binary_logical_judge_calls += int(
            binary_schema.get("logical_binary_judge_calls") or 0
        )
        binary_response_attempts += int(
            binary_schema.get("response_attempts") or 0
        )
        binary_schema_repair_retries += int(
            binary_schema.get("schema_repair_retries") or 0
        )
        binary_schema_repair_recoveries += int(
            binary_schema.get("schema_repair_recoveries") or 0
        )
        binary_schema_repair_failures += int(
            binary_schema.get("schema_repair_failures") or 0
        )
        latency = record.get("elapsed_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        comparison_path = Path(str(record["scene_comparison_path"]))
        comparison = read_json(comparison_path)
        if record.get("grouping_status") not in {None, "complete"}:
            grouping_failures += 1
        report = read_json(Path(str(record["scene_quality_report_path"])))
        report_metrics = report.get("metrics")
        report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
        control = (
            read_json(Path(str(record["control_manifest_path"])))
            if record.get("control_manifest_path")
            else {}
        )
        telemetry = telemetry_by_metric(control)
        comparisons = comparison.get("metrics")
        comparisons = comparisons if isinstance(comparisons, dict) else {}
        for metric in metrics:
            item = comparisons.get(metric)
            item = item if isinstance(item, dict) else {}
            human = item.get("human")
            human = human if isinstance(human, dict) else {}
            model = item.get("model")
            model = model if isinstance(model, dict) else {}
            summary = metric_summaries[metric]
            if record.get("final_decision_status") == "unresolved":
                summary["diagnostic_only_cases"] += 1
            expected = human.get("expected")
            predicted = model.get("prediction")
            if expected in {"valid", "invalid"}:
                summary["human_distribution"][expected] += 1
            if predicted in {"valid", "invalid"}:
                summary["predicted_distribution"][predicted] += 1
                summary["evaluated"] += 1
            else:
                summary["unresolved"] += 1
            if human.get("unclear") is True:
                summary["excluded_unclear"] += 1
            if item.get("included_in_accuracy") is True:
                if item.get("matches") is True:
                    summary["correct"] += 1
                else:
                    summary["incorrect"] += 1
            metric_report = report_metrics.get(metric)
            metric_report = (
                metric_report if isinstance(metric_report, dict) else {}
            )
            failure_counts = metric_failure_counts(metric_report)
            summary["camera_render_failures"] += failure_counts[
                "camera_render_failures"
            ]
            summary["judge_failures"] += failure_counts["judge_failures"]
            metric_telemetry = telemetry.get(metric, {})
            for key in (
                "judge_calls",
                "vlm_selector_calls",
                "preview_image_count",
                "final_image_count",
                "evidence_repair_count",
                "evidence_recovery_count",
            ):
                summary[key] += int(metric_telemetry.get(key) or 0)
            summary["initial_image_count"] += initial_image_count(
                metric_report
            )
    for summary in metric_summaries.values():
        denominator = summary["correct"] + summary["incorrect"]
        summary["accuracy"] = (
            summary["correct"] / denominator if denominator else None
        )
        total_judge_calls += summary["judge_calls"]
        total_selector_calls += summary["vlm_selector_calls"]
        summary["grouping_failures"] = grouping_failures
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": (
            "complete"
            if successful == len(case_records)
            else "partial"
            if successful
            else "failed"
        ),
        "source_prompt_used": False,
        "elapsed_seconds": elapsed_seconds,
        "average_case_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "totals": {
            "cases": len(case_records),
            "successful": successful,
            "failed": len(case_records) - successful,
            "grouping_failures": grouping_failures,
            "final_unresolved": final_unresolved,
            "l1_engineering_failure_cases": (
                l1_engineering_failure_cases
            ),
            "binary_logical_judge_calls": binary_logical_judge_calls,
            "binary_response_attempts": (
                binary_response_attempts
            ),
            "binary_schema_repair_retries": (
                binary_schema_repair_retries
            ),
            "binary_schema_repair_recoveries": (
                binary_schema_repair_recoveries
            ),
            "binary_schema_repair_failures": (
                binary_schema_repair_failures
            ),
            "judge_calls": total_judge_calls,
            "vlm_camera_selector_calls": total_selector_calls,
        },
        "metrics": metric_summaries,
    }


def empty_metric_summary(*, total: int) -> dict[str, Any]:
    return {
        "total": total,
        "evaluated": 0,
        "unresolved": 0,
        "excluded_unclear": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy": None,
        "predicted_distribution": {"valid": 0, "invalid": 0},
        "human_distribution": {"valid": 0, "invalid": 0},
        "grouping_failures": 0,
        "diagnostic_only_cases": 0,
        "case_failures": 0,
        "camera_render_failures": 0,
        "judge_failures": 0,
        "judge_calls": 0,
        "vlm_selector_calls": 0,
        "initial_image_count": 0,
        "final_image_count": 0,
        "preview_image_count": 0,
        "evidence_repair_count": 0,
        "evidence_recovery_count": 0,
    }


def telemetry_by_metric(control_manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    integration = control_manifest.get("integration")
    integration = integration if isinstance(integration, dict) else {}
    runtime = integration.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    calls = runtime.get("controlled_calls")
    calls = calls if isinstance(calls, list) else []
    result: dict[str, dict[str, int]] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        metric = str(call.get("metric") or "")
        if not metric:
            continue
        target = result.setdefault(
            metric,
            {
                "judge_calls": 0,
                "vlm_selector_calls": 0,
                "preview_image_count": 0,
                "final_image_count": 0,
                "evidence_repair_count": 0,
                "evidence_recovery_count": 0,
            },
        )
        audit = call.get("audit")
        audit = audit if isinstance(audit, dict) else {}
        telemetry = audit.get("experiment_telemetry")
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        target["judge_calls"] += int(telemetry.get("judge_calls") or 0)
        target["vlm_selector_calls"] += int(
            telemetry.get("vlm_selector_calls") or 0
        )
        target["preview_image_count"] += int(
            telemetry.get("preview_render_count") or 0
        )
        target["final_image_count"] += int(
            telemetry.get("final_render_count") or 0
        )
        if int(audit.get("rounds_used") or 0) > 0:
            target["evidence_repair_count"] += 1
        evaluation = audit.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        if evaluation.get("evidence_recovery_outcome") in {
            "recovered",
            "recovered_after_repair",
        }:
            target["evidence_recovery_count"] += 1
    return result


def initial_image_count(metric_report: dict[str, Any]) -> int:
    group_results = metric_report.get("group_results")
    if isinstance(group_results, list):
        return sum(
            len(item.get("evidence_paths") or [])
            for item in group_results
            if isinstance(item, dict)
        )
    paths = metric_report.get("evidence_paths")
    return len(paths) if isinstance(paths, list) else 0


def metric_failure_counts(metric_report: dict[str, Any]) -> dict[str, int]:
    camera_failures = 0
    judge_failures = 0
    group_results = metric_report.get("group_results")
    group_results = group_results if isinstance(group_results, list) else []
    for item in group_results:
        if not isinstance(item, dict):
            continue
        resolution = item.get("evidence_resolution")
        resolution = resolution if isinstance(resolution, dict) else {}
        if resolution.get("provider_status") == "failed":
            camera_failures += 1
        if item.get("reason") == "vlm_judge_failed":
            judge_failures += 1
    if metric_report.get("reason") == "vlm_judge_failed":
        judge_failures += 1
    return {
        "camera_render_failures": camera_failures,
        "judge_failures": judge_failures,
    }


def safe_route_manifest(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "authorization_configured": bool(
            route.get("authorization_configured")
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
