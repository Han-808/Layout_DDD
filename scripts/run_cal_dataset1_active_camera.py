#!/usr/bin/env python3
"""Run a judge-free active-camera pilot over cal_dataset1.

The runner reuses the frozen materialized ``.blend`` scenes from exp1_1 and
exp1_1_fine_edge, but never reuses their rendered evidence.  It compares four
camera/selector policies while keeping the current metric VisualConfig,
``local`` candidate generator, renderer, event universe, and detector
records frozen:

* ``deterministic_current``: current per-metric deterministic camera policy;
* ``static_vlm_topk``: one selector call over a frozen candidate bank;
* ``bounded_query_cov_all``: unconditional bounded selector/action loop;
* ``conditional_active_shadow``: deterministic packet first, active repair
  only when the sufficiency gate routes it, with official output unchanged.

There is deliberately no final metric judgement in this program.  Its outputs
measure evidence sufficiency, selector/action cost, failures, and trajectory
provenance.  Semantic labels are retained only for stratification; ambiguous
events are never silently treated as binary ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering import BlenderRenderer, CYCLES_DEVICES, RENDER_ENGINES
from benchmark.rendering.camera_pose import DEFAULT_CAMERA_CANDIDATE_POLICY
from benchmark.visual_judge.active_fallback import (
    build_conditional_active_camera_evidence_provider,
)
from benchmark.visual_judge.evidence_sufficiency import (
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.openai_compatible import (
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.p0b import build_p0b_local_evidence_request
from benchmark.visual_judge.render_views import CameraEvidenceProvider
from benchmark.visual_judge.visual_config import (
    DEFAULT_P0B_VISUAL_CONFIGS,
    compose_default_p0b_visual_evidence,
)


_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_cal_dataset1_camera_evidence_helpers",
    PROJECT_ROOT / "scripts" / "run_cal_dataset1_camera_evidence.py",
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load cal_dataset1 camera-evidence helpers")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


SCHEMA_VERSION = "cal_dataset1_active_camera_experiment_v1"
RESULT_SCHEMA_VERSION = "cal_dataset1_active_camera_event_result_v1"
SUMMARY_SCHEMA_VERSION = "cal_dataset1_active_camera_summary_v1"
SPLITS = ("obvious_distortion", "subtle_distortion", "fine_edge")
METRICS = ("collision", "oob", "support")
ARMS = (
    "deterministic_current",
    "static_vlm_topk",
    "bounded_query_cov_all",
    "conditional_active_shadow",
)
SELECTOR_ARMS = frozenset(ARMS[1:])
ARM_DESCRIPTIONS = {
    "deterministic_current": (
        "Current per-metric deterministic camera policy and current fixed-shape "
        "VisualConfig; no selector call."
    ),
    "static_vlm_topk": (
        "VLM selects the current metric-specific local-view budget from one "
        "frozen local candidate bank; max_steps=0."
    ),
    "bounded_query_cov_all": (
        "Every event runs the new metric-specific proposal/recheck loop from an "
        "explicit experimental camera-repair control, independent of the "
        "deterministic trigger gate and without producing a metric verdict."
    ),
    "conditional_active_shadow": (
        "Current deterministic packet is official; camera-repairable "
        "insufficiency may run bounded active repair as a shadow counterfactual."
    ),
}
DEFAULT_MATERIALIZED_ROOTS = (
    PROJECT_ROOT / "Support" / "artifacts" / "outputs" / "exp1_1",
    PROJECT_ROOT / "Support" / "artifacts" / "outputs" / "exp1_1_fine_edge",
)
DEFAULT_SELECTOR_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "gpt5_6_sol_litellm_local_visual_config_judge.json"
)
IMPLEMENTATION_FILES = (
    "src/benchmark/rendering/blender.py",
    "src/benchmark/rendering/blender_camera_worker.py",
    "src/benchmark/rendering/blender_collision_mask_worker.py",
    "src/benchmark/rendering/blender_collision_overlay_worker.py",
    "src/benchmark/rendering/blender_focus_bundle_worker.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/rendering/collision_overlay.py",
    "src/benchmark/rendering/segmentation_contour.py",
    "src/benchmark/visual_judge/active_fallback.py",
    "src/benchmark/visual_judge/active_policy.py",
    "src/benchmark/visual_judge/evidence_sufficiency.py",
    "src/benchmark/visual_judge/openai_compatible.py",
    "src/benchmark/visual_judge/render_views.py",
    "src/benchmark/visual_judge/visual_config.py",
    "scripts/run_cal_dataset1_active_camera.py",
)
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    materialized_roots = [
        Path(value).expanduser().resolve() for value in args.materialized_root
    ]
    cases = discover_experiment_cases(
        dataset_root,
        materialized_roots=materialized_roots,
        splits=set(args.split),
        metrics=set(args.metric),
        case_ids=set(args.case_id),
        max_cases=args.max_cases,
    )
    selector_config = (
        _read_selector_config(Path(args.selector_config).expanduser().resolve())
        if set(args.arm) & SELECTOR_ARMS
        else None
    )
    validate_selector_capacity(
        arms=tuple(args.arm),
        candidate_count=args.candidate_count,
        selector_config=selector_config,
        allow_missing=args.phase in {"plan", "aggregate"} or args.plan_only,
    )
    plan = build_plan(
        args=args,
        dataset_root=dataset_root,
        materialized_roots=materialized_roots,
        cases=cases,
        selector_config=selector_config,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only or args.phase == "plan":
        print(json.dumps(plan["counts"], indent=2))
        return

    if args.phase in {"prepare", "all"}:
        started = time.time()
        _prepare(
            args=args,
            out_dir=out_dir,
            cases=cases,
            selector_config=selector_config,
        )
        _write_json(
            out_dir / "run_manifest.json",
            build_run_manifest(
                out_dir=out_dir,
                plan=plan,
                experiment_id=args.experiment_id,
                elapsed_seconds=time.time() - started,
            ),
        )
    if args.phase in {"aggregate", "all", "prepare"}:
        summary = aggregate_results(out_dir, expected_plan=plan)
        print(json.dumps(summary["counts"], indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"),
    )
    parser.add_argument(
        "--materialized-root",
        action="append",
        default=[],
        help=(
            "Existing experiment root containing cases/<case>/scene/scene.blend. "
            "Repeat for multiple roots."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "cal_dataset1_active_camera"
        ),
    )
    parser.add_argument("--experiment-id", default="cal_dataset1_active_camera")
    parser.add_argument(
        "--phase",
        choices=("plan", "prepare", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--split", action="append", choices=SPLITS, default=[])
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--arm", action="append", choices=ARMS, default=[])
    parser.add_argument(
        "--selector-config",
        default=str(DEFAULT_SELECTOR_CONFIG),
        help="OpenAI-compatible model config used only through select_camera_views.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        choices=range(2, 9),
        default=5,
        help=(
            "Frozen candidate-bank size for every arm. Default 5 matches the "
            "checked-in selector max_images=5; larger values fail unless the "
            "selector config explicitly permits them."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        choices=range(0, 4),
        default=2,
        help="Maximum camera actions for bounded active arms.",
    )
    parser.add_argument(
        "--blender-bin",
        default="/Applications/Blender.app/Contents/MacOS/Blender",
    )
    parser.add_argument("--blender-timeout-seconds", type=int, default=1800)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument(
        "--render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--cycles-device", choices=CYCLES_DEVICES, default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preview-render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=256)
    parser.add_argument("--preview-cycles-samples", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.materialized_root:
        args.materialized_root = [str(path) for path in DEFAULT_MATERIALIZED_ROOTS]
    if not args.split:
        args.split = list(SPLITS)
    if not args.metric:
        args.metric = list(METRICS)
    if not args.arm:
        args.arm = list(ARMS)
    return args


def discover_experiment_cases(
    dataset_root: Path,
    *,
    materialized_roots: list[Path],
    splits: set[str],
    metrics: set[str],
    case_ids: set[str] | None = None,
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    """Resolve the frozen routed event universe and exactly one blend per case."""

    dataset_root = dataset_root.expanduser().resolve()
    blend_index = _materialized_blend_index(materialized_roots)
    selected = _HELPERS._selected_events(
        dataset_root,
        splits=splits,
        metrics=metrics,
        case_ids=case_ids or set(),
        max_cases=max_cases,
    )
    result: list[dict[str, Any]] = []
    for case in selected:
        case_id = str(case["case_id"])
        matches = blend_index.get(case_id, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"case {case_id!r} requires exactly one materialized scene.blend; "
                f"found {len(matches)}: {[str(path) for path in matches]}"
            )
        fixture = Path(str(case["fixture"])).resolve()
        report_path = (
            dataset_root
            / "evaluation"
            / "mesh"
            / case_id
            / "generic_validity.json"
        )
        geometry_path = (
            dataset_root
            / "evaluation"
            / "mesh_geometry"
            / case_id
            / "renders"
            / "collision_geometry_manifest.json"
        )
        provenance_path = matches[0].parent / "materialization_provenance.json"
        for required in (
            fixture / "generated_scene.json",
            fixture / "scene_request.json",
            fixture / "object_plan.json",
            fixture / "event_gt.json",
            fixture / "review.json",
            report_path,
            geometry_path,
            provenance_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"frozen active-camera input is missing for {case_id}: {required}"
                )
        review = _read_json(fixture / "review.json")
        value = deepcopy(case)
        value.update(
            {
                "blend_file": str(matches[0]),
                "materialization_provenance": str(provenance_path),
                "report_path": str(report_path),
                "collision_geometry_path": str(geometry_path),
                "review_status": review.get("status"),
                "reviewer": review.get("reviewer"),
            }
        )
        for event in value["events"]:
            event["scoring_eligible"] = _scoring_eligible(
                event,
                review_status=str(review.get("status") or ""),
            )
        result.append(value)
    return result


def _materialized_blend_index(
    materialized_roots: list[Path],
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for root in materialized_roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"materialized experiment root is missing: {root}")
        for path in sorted((resolved_root / "cases").glob("*/scene/scene.blend")):
            case_id = path.parents[1].name
            result.setdefault(case_id, []).append(path.resolve())
    return result


def _scoring_eligible(
    event: dict[str, Any],
    *,
    review_status: str,
) -> bool:
    label = str(event.get("semantic_label") or "").strip().lower()
    gt_basis = str(event.get("gt_basis") or "").strip().lower()
    approved = review_status.startswith("approved_")
    return bool(
        approved
        and label in {"valid", "invalid"}
        and gt_basis
        and gt_basis != "human_reviewed_edge_case"
    )


def validate_selector_capacity(
    *,
    arms: tuple[str, ...],
    candidate_count: int,
    selector_config: dict[str, Any] | None,
    allow_missing: bool,
) -> None:
    if not set(arms) & SELECTOR_ARMS:
        return
    if selector_config is None:
        if allow_missing:
            return
        raise ValueError("selector arms require --selector-config")
    max_images = int(selector_config.get("max_images") or 0)
    if max_images <= 0:
        raise ValueError(
            "selector config must freeze a positive max_images; implicit truncation "
            "of candidate previews is forbidden"
        )
    if int(candidate_count) > max_images:
        raise ValueError(
            f"candidate_count={candidate_count} exceeds selector max_images={max_images}; "
            "increase max_images or explicitly freeze a smaller candidate bank"
        )


def build_plan(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    materialized_roots: list[Path],
    cases: list[dict[str, Any]],
    selector_config: dict[str, Any] | None,
) -> dict[str, Any]:
    events = [event for case in cases for event in case["events"]]
    selector_identity = _selector_identity(selector_config)
    counts = {
        "cases": len(cases),
        "events": len(events),
        "arms": len(args.arm),
        "event_arm_runs": len(events) * len(args.arm),
        "events_by_metric": {
            metric: sum(str(event.get("metric")) == metric for event in events)
            for metric in METRICS
        },
        "cases_by_split": {
            split: sum(str(case.get("split")) == split for case in cases)
            for split in SPLITS
        },
        "events_by_split": {
            split: sum(
                len(case["events"])
                for case in cases
                if str(case.get("split")) == split
            )
            for split in SPLITS
        },
        "events_by_semantic_label": {
            label: sum(str(event.get("semantic_label")) == label for event in events)
            for label in ("invalid", "ambiguous")
        },
        "scoring_eligible": sum(
            bool(event.get("scoring_eligible")) for event in events
        ),
        "non_scoring": sum(
            not bool(event.get("scoring_eligible")) for event in events
        ),
    }
    plan = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": str(args.experiment_id),
        "experiment_type": "judge_free_camera_selector_and_repair_pilot",
        "dataset_root": str(dataset_root),
        "materialized_roots": [str(path) for path in materialized_roots],
        "final_metric_judge": "disabled",
        "metric_verdicts_produced": False,
        "accuracy_claim_supported": False,
        "controlled_variable": "camera_selector_and_bounded_repair_policy",
        "arms": {
            arm: ARM_DESCRIPTIONS[arm]
            for arm in args.arm
        },
        "frozen": {
            "candidate_policy": DEFAULT_CAMERA_CANDIDATE_POLICY,
            "candidate_count": int(args.candidate_count),
            "visual_config": deepcopy(DEFAULT_P0B_VISUAL_CONFIGS),
            "renderer": {
                "engine": args.render_engine,
                "size": [args.render_width, args.render_height],
                "preview_engine": args.preview_render_engine,
                "preview_size": [args.preview_width, args.preview_height],
            },
            "maximum_active_actions": int(args.max_steps),
            "selector_identity": selector_identity,
            "selector_candidate_count_validation": (
                "candidate_count_lte_max_images"
                if selector_identity is not None
                else "not_required_for_selected_arms"
                if not set(args.arm) & SELECTOR_ARMS
                else "deferred_plan_only"
            ),
            "scene_materialization": "reuse_blend_only",
            "rendered_evidence": "regenerate_under_current_code",
        },
        "label_policy": {
            "semantic_label_preserved": True,
            "binary_scoring_requires_frozen_valid_or_invalid_gt": True,
            "ambiguous_scoring_eligible": False,
            "labels_are_not_sent_to_selector": True,
        },
        "counts": counts,
        "cases": [
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "blend_file": case["blend_file"],
                "events": [
                    {
                        "metric": event["metric"],
                        "event_id": event["event_id"],
                        "semantic_label": event.get("semantic_label"),
                        "gt_basis": event.get("gt_basis"),
                        "scoring_eligible": bool(event["scoring_eligible"]),
                    }
                    for event in case["events"]
                ],
            }
            for case in cases
        ],
    }
    plan["plan_sha256"] = _json_sha256(plan)
    return plan


def _prepare(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    cases: list[dict[str, Any]],
    selector_config: dict[str, Any] | None,
) -> None:
    selector = (
        build_openai_compatible_vlm_judge(selector_config)
        if set(args.arm) & SELECTOR_ARMS and selector_config is not None
        else None
    )
    renderer = BlenderRenderer(
        blender_bin=Path(args.blender_bin),
        timeout_seconds=args.blender_timeout_seconds,
        width=args.render_width,
        height=args.render_height,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
        preview_render_engine=args.preview_render_engine,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        preview_cycles_samples=args.preview_cycles_samples,
    )
    total = sum(len(case["events"]) for case in cases) * len(args.arm)
    index = 0
    for case in cases:
        scene = _read_json(Path(case["fixture"]) / "generated_scene.json")
        report = _read_json(Path(case["report_path"]))
        geometry = _HELPERS._load_collision_geometry(
            Path(case["collision_geometry_path"])
        )
        prompt, relationships = _HELPERS._prompt_context(Path(case["fixture"]))
        for event in case["events"]:
            record = _HELPERS._report_record(
                report,
                str(event["metric"]),
                str(event["event_id"]),
                list(event.get("object_ids") or []),
            )
            detector, event_payload = _HELPERS._event_context(
                str(event["metric"]),
                record,
                event,
            )
            request = build_p0b_local_evidence_request(
                metric=str(event["metric"]),
                event=event_payload,
                prompt=prompt,
                relationships=relationships,
                scene=scene,
                detector_evidence=detector,
                object_ids=list(event.get("object_ids") or []),
            )
            for arm in args.arm:
                index += 1
                print(
                    f"[{index}/{total}] {case['case_id']} "
                    f"{event['metric']}:{event['event_id']} {arm}",
                    flush=True,
                )
                try:
                    _run_event_arm(
                        args=args,
                        out_dir=out_dir,
                        case=case,
                        event=event,
                        arm=arm,
                        request=request,
                        renderer=renderer,
                        selector=selector,
                        collision_geometry=geometry,
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    _write_event_error(
                        out_dir=out_dir,
                        case=case,
                        event=event,
                        arm=arm,
                        error=f"{type(exc).__name__}: {exc}",
                    )


def build_run_manifest(
    *,
    out_dir: Path,
    plan: dict[str, Any],
    experiment_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    completed_results = sorted(
        (out_dir / "cases").glob("*/events/*/arms/*/result.json")
    )
    result_hashes = {
        str(path.relative_to(out_dir)): _file_sha256(path)
        for path in completed_results
    }
    result_payloads = [_read_json(path) for path in completed_results]
    expected = int(plan["counts"]["event_arm_runs"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "plan_sha256": plan["plan_sha256"],
        "phase": "prepare",
        "final_metric_judge": "disabled",
        "metric_verdicts_produced": False,
        "expected_event_arm_runs": expected,
        "result_file_count": len(completed_results),
        "complete_result_count": sum(
            value.get("complete") is True for value in result_payloads
        ),
        "failed_result_count": sum(
            value.get("complete") is not True or bool(value.get("error"))
            for value in result_payloads
        ),
        "missing_result_count": max(0, expected - len(completed_results)),
        "experiment_plan_sha256": _file_sha256(
            out_dir / "experiment_plan.json"
        ),
        "result_sha256": result_hashes,
        "result_index_sha256": _json_sha256(result_hashes),
        "elapsed_seconds": elapsed_seconds,
    }


def _run_event_arm(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    case: dict[str, Any],
    event: dict[str, Any],
    arm: str,
    request: dict[str, Any],
    renderer: BlenderRenderer,
    selector: Any,
    collision_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    metric = str(event["metric"])
    event_dir = _event_arm_dir(out_dir, case, event, arm)
    provider_request = _provider_request_for_arm(arm, request)
    invocation_key = _evidence_invocation_key(
        args=args,
        case=case,
        event=event,
        arm=arm,
        provider_request=provider_request,
    )
    evidence_root = (
        event_dir
        / "evidence_invocations"
        / invocation_key
    )
    provider = _provider_for_arm(
        arm=arm,
        metric=metric,
        renderer=renderer,
        selector=selector,
        blend_file=Path(case["blend_file"]),
        out_dir=evidence_root,
        candidate_count=args.candidate_count,
        max_steps=args.max_steps,
        collision_geometry=collision_geometry,
    )
    source_contract = _source_contract(
        args=args,
        case=case,
        event=event,
        request=provider_request,
        arm=arm,
        provider=provider,
        evidence_invocation_key=invocation_key,
        evidence_root=evidence_root,
    )
    result_path = event_dir / "result.json"
    if args.resume and _result_ready(result_path, source_contract):
        print("  cached", flush=True)
        return _read_json(result_path)

    started = time.time()
    items = list(provider(provider_request))
    assessment = assess_visual_evidence_sufficiency(
        metric,
        items,
        request=request,
    )
    selected_items: list[dict[str, Any]] = []
    visual_config: dict[str, Any] | None = None
    visual_config_error: str | None = None
    try:
        selected_items, visual_config = compose_default_p0b_visual_evidence(
            metric,
            [item for item in items if isinstance(item, dict)],
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        visual_config_error = f"{type(exc).__name__}: {exc}"
    camera_manifests = _artifact_records(
        evidence_root,
        "camera_evidence_manifest.json",
    )
    fallback_manifests = _artifact_records(
        evidence_root,
        "active_camera_fallback_manifest.json",
    )
    shadow = _shadow_summary(
        arm=arm,
        official_assessment=assessment,
        fallback_manifests=fallback_manifests,
    )
    cost = _cost_summary(
        camera_manifests,
        fallback_manifests,
        elapsed_seconds=time.time() - started,
    )
    failure_attribution = _failure_attribution(
        official_assessment=assessment,
        visual_config_error=visual_config_error,
        shadow=shadow,
        execution_error=None,
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "complete": True,
        "case_id": case["case_id"],
        "split": case["split"],
        "metric": metric,
        "event_id": event["event_id"],
        "object_ids": list(event.get("object_ids") or []),
        "semantic_label": event.get("semantic_label"),
        "gt_basis": event.get("gt_basis"),
        "scoring_eligible": bool(event.get("scoring_eligible")),
        "metric_verdict": None,
        "final_metric_judge": "disabled",
        "arm": arm,
        "camera_selection_control": deepcopy(
            provider_request.get("_camera_evidence_deficiency")
        ),
        "evidence_invocation_key": invocation_key,
        "evidence_root": str(evidence_root.resolve()),
        "source_contract": source_contract,
        "source_contract_sha256": _json_sha256(source_contract),
        "policy": deepcopy(getattr(provider, "policy_config", None)),
        "evidence_sufficiency": assessment,
        "visual_config": visual_config,
        "visual_config_error": visual_config_error,
        "selected_packet": _evidence_records(selected_items),
        "all_provider_items": _evidence_records(
            [item for item in items if isinstance(item, dict)]
        ),
        "camera_manifests": camera_manifests,
        "active_fallback_manifests": fallback_manifests,
        "shadow": shadow,
        "cost": cost,
        "failure_attribution": failure_attribution,
    }
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_path, result)
    return result


def _provider_for_arm(
    *,
    arm: str,
    metric: str,
    renderer: BlenderRenderer,
    selector: Any,
    blend_file: Path,
    out_dir: Path,
    candidate_count: int,
    max_steps: int,
    collision_geometry: dict[str, Any] | None,
) -> Callable[[dict[str, Any]], list[Any]]:
    local_view_count = int(
        DEFAULT_P0B_VISUAL_CONFIGS[metric]["local_view_count"]
    )
    shared = {
        "renderer": renderer,
        "blend_file": blend_file,
        "out_dir": out_dir,
        "max_views": local_view_count,
        "candidate_count": candidate_count,
        "collision_overlay": True,
        "collision_contour": True,
        "collision_geometry": collision_geometry,
        "highlighted_global_pose_policy": "global_top",
        "candidate_policy": DEFAULT_CAMERA_CANDIDATE_POLICY,
    }
    if arm == "deterministic_current":
        return CameraEvidenceProvider(
            **shared,
            mode="auto",
            selector=None,
            max_steps=0,
        )
    if selector is None:
        raise ValueError(f"arm {arm} requires a selector config")
    if arm == "static_vlm_topk":
        return CameraEvidenceProvider(
            **shared,
            mode="query_cov",
            selector=selector,
            max_steps=0,
        )
    if arm == "bounded_query_cov_all":
        return CameraEvidenceProvider(
            **shared,
            mode="query_cov",
            selector=selector,
            max_steps=max_steps,
            active_repair=True,
        )
    if arm == "conditional_active_shadow":
        kwargs: dict[str, Any] = {
            **shared,
            "deterministic_mode": "auto",
            "selector": selector,
            "max_steps": max_steps,
            "fail_on_exhausted": False,
        }
        signature = inspect.signature(
            build_conditional_active_camera_evidence_provider
        )
        if "shadow_mode" not in signature.parameters:
            raise RuntimeError(
                "conditional_active_shadow requires the active fallback "
                "implementation to expose shadow_mode"
            )
        kwargs["shadow_mode"] = True
        return build_conditional_active_camera_evidence_provider(**kwargs)
    raise ValueError(f"unsupported active-camera arm {arm!r}")


def _provider_request_for_arm(
    arm: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Add only the control signal required by the unconditional repair arm."""

    value = deepcopy(request)
    if arm != "bounded_query_cov_all":
        return value
    metric = str(request.get("metric") or "").strip().lower()
    value["_camera_selection_phase"] = "active_fallback"
    value["_camera_evidence_deficiency"] = {
        "schema_version": "experimental_unconditional_camera_repair_v1",
        "metric": metric,
        # This is an experiment-control condition, not a measured scene error
        # and not a metric verdict.  The active loop immediately replaces it
        # with its measured preview sufficiency after the first selector call.
        "status": "experimental_forced_probe",
        "metric_verdict": None,
        "experimental_ablation": True,
        "source": "bounded_query_cov_all_unconditional_arm",
        "camera_repairable": True,
        "repairability": "camera",
        "trigger_recommended": True,
        "reason_codes": ["measured_local_visibility_insufficient"],
        "deficiencies": [
            {
                "code": "measured_local_visibility_insufficient",
                "repairability": "camera",
                "basis": "experimental_initial_repair_control",
            }
        ],
    }
    return value


def _evidence_invocation_key(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    event: dict[str, Any],
    arm: str,
    provider_request: dict[str, Any],
) -> str:
    """Content-address one provider invocation before discovering artifacts."""

    fixture = Path(case["fixture"])
    selector_config_path = Path(args.selector_config).expanduser().resolve()
    payload = {
        "schema_version": "cal_dataset1_evidence_invocation_v1",
        "arm": arm,
        "event": {
            "case_id": case["case_id"],
            "metric": event["metric"],
            "event_id": event["event_id"],
        },
        "provider_request_sha256": _json_sha256(provider_request),
        "blend_file_sha256": _file_sha256(Path(case["blend_file"])),
        "input_sha256": {
            "generated_scene": _file_sha256(
                fixture / "generated_scene.json"
            ),
            "event_gt": _file_sha256(fixture / "event_gt.json"),
            "detector_report": _file_sha256(Path(case["report_path"])),
            "collision_geometry": _file_sha256(
                Path(case["collision_geometry_path"])
            ),
        },
        "policy": {
            "candidate_policy": DEFAULT_CAMERA_CANDIDATE_POLICY,
            "candidate_count": int(args.candidate_count),
            "maximum_active_actions": int(args.max_steps),
            "visual_config": DEFAULT_P0B_VISUAL_CONFIGS[
                str(event["metric"])
            ],
        },
        "renderer": {
            "render_engine": args.render_engine,
            "render_size": [args.render_width, args.render_height],
            "preview_render_engine": args.preview_render_engine,
            "preview_size": [args.preview_width, args.preview_height],
        },
        "selector_config_sha256": (
            _file_sha256(selector_config_path)
            if arm in SELECTOR_ARMS and selector_config_path.is_file()
            else None
        ),
        "implementation_sha256": {
            path: _file_sha256(PROJECT_ROOT / path)
            for path in IMPLEMENTATION_FILES
            if (PROJECT_ROOT / path).is_file()
        },
    }
    return _json_sha256(payload)[:24]


def _source_contract(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    event: dict[str, Any],
    request: dict[str, Any],
    arm: str,
    provider: Any,
    evidence_invocation_key: str,
    evidence_root: Path,
) -> dict[str, Any]:
    fixture = Path(case["fixture"])
    paths = {
        "generated_scene": fixture / "generated_scene.json",
        "event_gt": fixture / "event_gt.json",
        "scene_request": fixture / "scene_request.json",
        "object_plan": fixture / "object_plan.json",
        "detector_report": Path(case["report_path"]),
        "collision_geometry": Path(case["collision_geometry_path"]),
        "materialization_provenance": Path(case["materialization_provenance"]),
    }
    return {
        "schema_version": "cal_dataset1_active_camera_source_contract_v1",
        "arm": arm,
        "evidence_invocation_key": evidence_invocation_key,
        "evidence_root": str(evidence_root.resolve()),
        "event_identity": {
            "case_id": case["case_id"],
            "metric": event["metric"],
            "event_id": event["event_id"],
            "object_ids": list(event.get("object_ids") or []),
        },
        "input_sha256": {
            key: _file_sha256(path)
            for key, path in paths.items()
        },
        "request_sha256": _json_sha256(request),
        "blend_file": str(Path(case["blend_file"]).resolve()),
        "blend_file_sha256": _provider_blend_sha256(provider),
        "provider_policy": deepcopy(getattr(provider, "policy_config", None)),
        "candidate_policy": DEFAULT_CAMERA_CANDIDATE_POLICY,
        "candidate_count": int(args.candidate_count),
        "maximum_active_actions": int(args.max_steps),
        "implementation_sha256": {
            path: _file_sha256(PROJECT_ROOT / path)
            for path in IMPLEMENTATION_FILES
            if (PROJECT_ROOT / path).is_file()
        },
    }


def _provider_blend_sha256(provider: Any) -> str:
    policy = getattr(provider, "policy_config", None)
    if isinstance(policy, dict):
        value = policy.get("source_blend_sha256")
        if value:
            return str(value)
        deterministic = policy.get("deterministic_policy")
        if isinstance(deterministic, dict) and deterministic.get(
            "source_blend_sha256"
        ):
            return str(deterministic["source_blend_sha256"])
    raise RuntimeError("camera provider did not expose a source blend content hash")


def _evidence_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"rendered camera evidence is missing: {path}")
        result.append(
            {
                "role": item.get("role"),
                "view_id": item.get("view_id")
                or (
                    item.get("pose", {}).get("id")
                    if isinstance(item.get("pose"), dict)
                    else None
                ),
                "path": str(path),
                "sha256": _file_sha256(path),
            }
        )
    return result


def _artifact_records(root: Path, filename: str) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "payload": _read_json(path),
        }
        for path in sorted(root.rglob(filename))
        if path.is_file()
    ]


def _shadow_summary(
    *,
    arm: str,
    official_assessment: dict[str, Any],
    fallback_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize the shadow counterfactual without changing official evidence."""

    if len(fallback_manifests) > 1:
        raise RuntimeError(
            "one event arm produced multiple active fallback manifests"
        )
    if fallback_manifests:
        payload = fallback_manifests[0]["payload"]
        before = (
            deepcopy(payload.get("deterministic_assessment"))
            if isinstance(payload.get("deterministic_assessment"), dict)
            else None
        )
        active_attempted = bool(payload.get("active_attempted"))
        after = (
            deepcopy(payload.get("final_assessment"))
            if active_attempted
            and isinstance(payload.get("final_assessment"), dict)
            else None
        )
        # A non-triggered v2 manifest has final==deterministic and therefore its
        # raw would_replace field may be true.  It is not a counterfactual
        # replacement unless an active attempt actually happened.
        counterfactual_would_replace = bool(
            active_attempted
            and payload.get("counterfactual_would_replace")
        )
        repair_success = bool(
            active_attempted
            and not payload.get("active_error")
            and isinstance(before, dict)
            and before.get("status") == "insufficient"
            and isinstance(after, dict)
            and after.get("status") == "sufficient"
        )
        return {
            "active_attempted": active_attempted,
            "counterfactual_would_replace": counterfactual_would_replace,
            "repair_success": repair_success,
            "deterministic_before_assessment": before,
            "counterfactual_after_assessment": after,
            "trigger_reason_codes": list(
                (before or {}).get("reason_codes") or []
            )
            if active_attempted
            else [],
            "active_error": payload.get("active_error"),
            "deterministic_error": payload.get("deterministic_error"),
            "official_packet_source": str(
                payload.get("official_packet_source") or "deterministic"
            ),
            "shadow_mode": bool(payload.get("shadow_mode")),
            "fallback_manifest": fallback_manifests[0]["path"],
        }
    source_by_arm = {
        "deterministic_current": "deterministic",
        "static_vlm_topk": "static_vlm_selection",
        "bounded_query_cov_all": "bounded_query_cov",
    }
    return {
        "active_attempted": False,
        "counterfactual_would_replace": False,
        "repair_success": False,
        "deterministic_before_assessment": (
            deepcopy(official_assessment)
            if arm == "deterministic_current"
            else None
        ),
        "counterfactual_after_assessment": None,
        "trigger_reason_codes": [],
        "active_error": None,
        "deterministic_error": None,
        "official_packet_source": source_by_arm.get(arm, "unknown"),
        "shadow_mode": False,
        "fallback_manifest": None,
    }


def _cost_summary(
    camera_manifests: list[dict[str, Any]],
    fallback_manifests: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    selector_calls = 0
    camera_actions = 0
    trajectories: list[dict[str, Any]] = []
    for artifact in camera_manifests:
        payload = artifact["payload"]
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            continue
        trajectory = (
            selection.get("trajectory")
            if isinstance(selection.get("trajectory"), list)
            else selection.get("steps")
            if isinstance(selection.get("steps"), list)
            else []
        )
        explicit_selector_calls = selection.get("selector_call_count")
        selector_calls += (
            int(explicit_selector_calls)
            if isinstance(explicit_selector_calls, int)
            and explicit_selector_calls >= 0
            else len(trajectory)
        )
        explicit_camera_actions = selection.get("camera_action_count")
        if (
            isinstance(explicit_camera_actions, int)
            and explicit_camera_actions >= 0
        ):
            camera_actions += explicit_camera_actions
        else:
            camera_actions += sum(
                _step_executed_action(step)
                for step in trajectory
                if isinstance(step, dict)
            )
        trajectories.append(
            {
                "manifest": artifact["path"],
                "selected_view_ids": selection.get("selected_view_ids"),
                "stop_reason": selection.get("stop_reason"),
                "trajectory": trajectory,
            }
        )
    active_used = any(
        bool(record["payload"].get("active_used"))
        for record in fallback_manifests
    )
    active_attempted = any(
        bool(record["payload"].get("active_attempted"))
        for record in fallback_manifests
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "selector_calls": selector_calls,
        "camera_actions": camera_actions,
        "active_used": active_used,
        "active_attempted": active_attempted,
        "trajectories": trajectories,
    }


def _step_executed_action(step: dict[str, Any]) -> int:
    execution = step.get("action_execution")
    if isinstance(execution, dict):
        return int(execution.get("executed") is True)
    decision = step.get("decision")
    if isinstance(decision, dict):
        return int(isinstance(decision.get("action"), dict))
    return int(isinstance(step.get("action"), dict))


def _failure_attribution(
    *,
    official_assessment: dict[str, Any] | None,
    visual_config_error: str | None,
    shadow: dict[str, Any] | None,
    execution_error: str | None,
) -> dict[str, Any]:
    """Separate execution failures, unresolved evidence, and shadow failures."""

    assessment = (
        official_assessment
        if isinstance(official_assessment, dict)
        else {}
    )
    shadow_value = shadow if isinstance(shadow, dict) else {}
    status = str(assessment.get("status") or "unknown")
    reason_codes = list(assessment.get("reason_codes") or [])
    repairability = assessment.get("repairability")
    if execution_error:
        return {
            "outcome": "execution_failed",
            "stage": _execution_failure_stage(execution_error),
            "code": "event_arm_execution_failed",
            "error": execution_error,
            "official_evidence_status": status,
            "reason_codes": reason_codes,
            "repairability": repairability,
            "official_packet_usable": False,
        }
    if visual_config_error:
        return {
            "outcome": "unresolved",
            "stage": "visual_config_composition",
            "code": "current_visual_config_incomplete",
            "error": visual_config_error,
            "official_evidence_status": status,
            "reason_codes": reason_codes,
            "repairability": repairability,
            "official_packet_usable": False,
        }
    if shadow_value.get("active_error"):
        return {
            "outcome": "shadow_counterfactual_failed",
            "stage": "active_camera_counterfactual",
            "code": "active_camera_execution_failed",
            "error": shadow_value.get("active_error"),
            "official_evidence_status": status,
            "reason_codes": reason_codes,
            "repairability": repairability,
            "official_packet_usable": status == "sufficient",
        }
    if status == "unknown":
        return {
            "outcome": "unresolved",
            "stage": "evidence_sufficiency",
            "code": "evidence_sufficiency_unknown",
            "error": None,
            "official_evidence_status": status,
            "reason_codes": reason_codes,
            "repairability": repairability,
            "official_packet_usable": False,
        }
    if status == "insufficient":
        code = (
            "camera_repairable_evidence_insufficient"
            if assessment.get("camera_repairable") is True
            else "non_camera_evidence_insufficient"
        )
        return {
            "outcome": "unresolved",
            "stage": "evidence_sufficiency",
            "code": code,
            "error": None,
            "official_evidence_status": status,
            "reason_codes": reason_codes,
            "repairability": repairability,
            "official_packet_usable": False,
        }
    return {
        "outcome": "resolved",
        "stage": None,
        "code": "none",
        "error": None,
        "official_evidence_status": status,
        "reason_codes": reason_codes,
        "repairability": repairability,
        "official_packet_usable": status == "sufficient",
    }


def _execution_failure_stage(error: str) -> str:
    normalized = error.lower()
    if "blender" in normalized or "render" in normalized:
        return "render"
    if (
        "endpoint" in normalized
        or "http" in normalized
        or "selector" in normalized
        or "model" in normalized
    ):
        return "selector"
    return "event_arm"


def _event_arm_dir(
    out_dir: Path,
    case: dict[str, Any],
    event: dict[str, Any],
    arm: str,
) -> Path:
    return (
        out_dir
        / "cases"
        / str(case["case_id"])
        / "events"
        / f"{event['metric']}__{_safe_name(str(event['event_id']))}"
        / "arms"
        / arm
    )


def _write_event_error(
    *,
    out_dir: Path,
    case: dict[str, Any],
    event: dict[str, Any],
    arm: str,
    error: str,
) -> None:
    event_dir = _event_arm_dir(out_dir, case, event, arm)
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        event_dir / "result.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "complete": False,
            "case_id": case["case_id"],
            "split": case["split"],
            "metric": event["metric"],
            "event_id": event["event_id"],
            "object_ids": list(event.get("object_ids") or []),
            "semantic_label": event.get("semantic_label"),
            "gt_basis": event.get("gt_basis"),
            "scoring_eligible": bool(event.get("scoring_eligible")),
            "metric_verdict": None,
            "final_metric_judge": "disabled",
            "arm": arm,
            "error": error,
            "shadow": {
                "active_attempted": False,
                "counterfactual_would_replace": False,
                "repair_success": False,
                "deterministic_before_assessment": None,
                "counterfactual_after_assessment": None,
                "trigger_reason_codes": [],
                "active_error": None,
                "deterministic_error": None,
                "official_packet_source": "unavailable",
                "shadow_mode": arm == "conditional_active_shadow",
                "fallback_manifest": None,
            },
            "cost": {
                "elapsed_seconds": None,
                "selector_calls": None,
                "camera_actions": None,
                "active_used": False,
                "active_attempted": False,
                "trajectories": [],
            },
            "failure_attribution": _failure_attribution(
                official_assessment=None,
                visual_config_error=None,
                shadow=None,
                execution_error=error,
            ),
        },
    )


def _result_ready(
    path: Path,
    expected_source_contract: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
        if (
            value.get("schema_version") != RESULT_SCHEMA_VERSION
            or value.get("complete") is not True
            or value.get("source_contract") != expected_source_contract
            or value.get("source_contract_sha256")
            != _json_sha256(expected_source_contract)
        ):
            return False
        for group in ("selected_packet", "all_provider_items"):
            for item in value.get(group) or []:
                evidence = Path(str(item.get("path") or ""))
                if (
                    not evidence.is_file()
                    or item.get("sha256") != _file_sha256(evidence)
                ):
                    return False
        for group in ("camera_manifests", "active_fallback_manifests"):
            for item in value.get(group) or []:
                artifact = Path(str(item.get("path") or ""))
                if (
                    not artifact.is_file()
                    or item.get("sha256") != _file_sha256(artifact)
                ):
                    return False
    except Exception:
        return False
    return True


def aggregate_results(
    out_dir: Path,
    *,
    expected_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "cases").glob("*/events/*/arms/*/result.json")):
        value = _read_json(path)
        assessment = (
            value.get("evidence_sufficiency")
            if isinstance(value.get("evidence_sufficiency"), dict)
            else {}
        )
        cost = value.get("cost") if isinstance(value.get("cost"), dict) else {}
        shadow = (
            value.get("shadow")
            if isinstance(value.get("shadow"), dict)
            else {}
        )
        before = (
            shadow.get("deterministic_before_assessment")
            if isinstance(
                shadow.get("deterministic_before_assessment"),
                dict,
            )
            else {}
        )
        after = (
            shadow.get("counterfactual_after_assessment")
            if isinstance(
                shadow.get("counterfactual_after_assessment"),
                dict,
            )
            else {}
        )
        attribution = (
            value.get("failure_attribution")
            if isinstance(value.get("failure_attribution"), dict)
            else _failure_attribution(
                official_assessment=assessment,
                visual_config_error=value.get("visual_config_error"),
                shadow=shadow,
                execution_error=value.get("error"),
            )
        )
        rows.append(
            {
                "case_id": value.get("case_id"),
                "split": value.get("split"),
                "metric": value.get("metric"),
                "event_id": value.get("event_id"),
                "semantic_label": value.get("semantic_label"),
                "scoring_eligible": bool(value.get("scoring_eligible")),
                "arm": value.get("arm"),
                "status": assessment.get("status"),
                "repairability": assessment.get("repairability"),
                "trigger_recommended": assessment.get("trigger_recommended"),
                "active_attempted": bool(shadow.get("active_attempted")),
                "active_used": cost.get("active_used"),
                "counterfactual_would_replace": bool(
                    shadow.get("counterfactual_would_replace")
                ),
                "repair_success": bool(shadow.get("repair_success")),
                "before_status": before.get("status"),
                "after_status": after.get("status"),
                "trigger_reason_codes": json.dumps(
                    shadow.get("trigger_reason_codes") or [],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "active_error": shadow.get("active_error"),
                "official_packet_source": shadow.get(
                    "official_packet_source"
                ),
                "selector_calls": cost.get("selector_calls"),
                "camera_actions": cost.get("camera_actions"),
                "elapsed_seconds": cost.get("elapsed_seconds"),
                "failure_outcome": attribution.get("outcome"),
                "failure_stage": attribution.get("stage"),
                "failure_code": attribution.get("code"),
                "failure_error": attribution.get("error"),
                "official_packet_usable": attribution.get(
                    "official_packet_usable"
                ),
                "complete": bool(value.get("complete")),
                "error": value.get("error"),
                "result_path": str(path.resolve()),
            }
        )
    arm_summaries = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        arm_summaries.append({
            "arm": arm,
            **_aggregate_group(arm_rows),
            "metrics": {
                metric: _aggregate_group(
                    [row for row in arm_rows if row["metric"] == metric]
                )
                for metric in METRICS
            },
        })
    expected_results = (
        int(expected_plan["counts"]["event_arm_runs"])
        if expected_plan is not None
        else None
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "final_metric_judge": "disabled",
        "metric_verdicts_produced": False,
        "accuracy_claim_supported": False,
        "plan_sha256": (
            expected_plan.get("plan_sha256")
            if expected_plan is not None
            else None
        ),
        "counts": {
            "results": len(rows),
            "expected_results": expected_results,
            "missing_results": (
                max(0, expected_results - len(rows))
                if expected_results is not None
                else None
            ),
            "complete": sum(row["complete"] for row in rows),
            "failures": sum(bool(row["error"]) for row in rows),
            "unresolved": sum(
                row["failure_outcome"] == "unresolved" for row in rows
            ),
            "shadow_counterfactual_failures": sum(
                row["failure_outcome"] == "shadow_counterfactual_failed"
                for row in rows
            ),
            "scoring_eligible": sum(row["scoring_eligible"] for row in rows),
            "ambiguous_non_scoring": sum(
                row["semantic_label"] == "ambiguous"
                and not row["scoring_eligible"]
                for row in rows
            ),
        },
        "arms": arm_summaries,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_summary_tsv(out_dir / "summary.tsv", rows)
    _write_failure_attribution_tsv(
        out_dir / "failure_attribution.tsv",
        rows,
    )
    return summary


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if row["active_attempted"]]
    repair_success_count = sum(row["repair_success"] for row in triggered)
    return {
        "result_count": len(rows),
        "complete_count": sum(row["complete"] for row in rows),
        "execution_failure_count": sum(bool(row["error"]) for row in rows),
        "unresolved_count": sum(
            row["failure_outcome"] == "unresolved" for row in rows
        ),
        "shadow_counterfactual_failure_count": sum(
            row["failure_outcome"] == "shadow_counterfactual_failed"
            for row in rows
        ),
        "official_status_counts": _counts(row["status"] for row in rows),
        "trigger_count": len(triggered),
        "trigger_reason_counts": _counts(
            reason
            for row in triggered
            for reason in _json_string_list(row["trigger_reason_codes"])
        ),
        "repair_success_count": repair_success_count,
        "repair_success_rate_on_triggered_subset": (
            repair_success_count / len(triggered)
            if triggered
            else None
        ),
        "counterfactual_would_replace_count": sum(
            row["counterfactual_would_replace"] for row in triggered
        ),
        "before_status_counts": _counts(
            row["before_status"]
            for row in rows
            if row["before_status"] is not None
        ),
        "after_status_counts": _counts(
            row["after_status"]
            for row in triggered
            if row["after_status"] is not None
        ),
        "selector_calls": sum(
            int(row["selector_calls"] or 0) for row in rows
        ),
        "camera_actions": sum(
            int(row["camera_actions"] or 0) for row in rows
        ),
        "elapsed_seconds": sum(
            float(row["elapsed_seconds"] or 0.0) for row in rows
        ),
        "official_packet_source_counts": _counts(
            row["official_packet_source"] for row in rows
        ),
        "failure_outcome_counts": _counts(
            row["failure_outcome"] for row in rows
        ),
        "failure_attribution_counts": _counts(
            row["failure_code"] for row in rows
        ),
    }


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return (
        [str(item) for item in parsed]
        if isinstance(parsed, list)
        else []
    )


def _write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "split",
        "metric",
        "event_id",
        "semantic_label",
        "scoring_eligible",
        "arm",
        "status",
        "repairability",
        "trigger_recommended",
        "active_attempted",
        "active_used",
        "counterfactual_would_replace",
        "repair_success",
        "before_status",
        "after_status",
        "trigger_reason_codes",
        "active_error",
        "official_packet_source",
        "selector_calls",
        "camera_actions",
        "elapsed_seconds",
        "failure_outcome",
        "failure_stage",
        "failure_code",
        "failure_error",
        "official_packet_usable",
        "complete",
        "error",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_failure_attribution_tsv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "case_id",
        "split",
        "metric",
        "event_id",
        "arm",
        "complete",
        "status",
        "failure_outcome",
        "failure_stage",
        "failure_code",
        "failure_error",
        "official_packet_usable",
        "active_attempted",
        "active_error",
        "official_packet_source",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_selector_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return None  # type: ignore[return-value]
    value = _read_json(path)
    if "api_key" in value:
        raise ValueError(
            "selector config must not contain a literal API key; use api_key_env"
        )
    return value


def _selector_identity(
    config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    allowed = (
        "provider",
        "name",
        "endpoint",
        "model",
        "api_key_env",
        "temperature",
        "max_tokens",
        "timeout_seconds",
        "response_format_json",
        "max_images",
        "max_context_chars",
    )
    return {
        key: deepcopy(config.get(key))
        for key in allowed
        if key in config
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _safe_name(value: str) -> str:
    return _HELPERS._safe_name(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    for stale in [
        key for key in _FILE_SHA256_CACHE if key[0] == str(resolved)
    ]:
        _FILE_SHA256_CACHE.pop(stale, None)
    _FILE_SHA256_CACHE[cache_key] = value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
