#!/usr/bin/env python3
"""Replay frozen cal_dataset1 images under controlled VisualConfig arms.

This is a judgement-only experiment. It derives image bundles from the
byte-frozen ``exp1_1`` global/local raw+highlight renders and never invokes
Blender, camera selection, detector recomputation, or generation. Ground truth
is attached only after each model call for scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.judge_cal_dataset1_camera_evidence import (  # noqa: E402
    _discover_manifests,
    _failed_result,
    _file_sha256,
    _judge_identity,
    _judge_one,
    _json_sha256,
    _read_json,
    _safe_name,
    _source_files,
    _verified_evidence,
    _verify_source_hashes,
    _write_json,
)


ARMS = (
    "fixed_global",
    "fixed_global_highlight",
    "presence_local_raw",
    "presence_local_raw_highlight",
    "presence_global_local_raw",
    "deterministic_metric_local",
    "order_local_first_full",
    "budget_global_first_compact",
    "budget_local_first_compact",
)
CONTOUR_ARM = "presence_local_raw_contour"
ALL_ARMS = (*ARMS, CONTOUR_ARM)
METRICS = ("collision", "oob", "support")
SEVERITIES = ("obvious", "subtle")
PAIR_DEFINITIONS = (
    ("global_highlight_effect", "fixed_global", "fixed_global_highlight"),
    ("local_highlight_effect", "presence_local_raw", "presence_local_raw_highlight"),
    (
        "local_contour_effect",
        "presence_local_raw",
        "presence_local_raw_contour",
    ),
    (
        "legacy_highlight_to_contour",
        "presence_local_raw_highlight",
        "presence_local_raw_contour",
    ),
    ("global_context_with_raw", "presence_local_raw", "presence_global_local_raw"),
    (
        "global_context_with_highlight",
        "presence_local_raw_highlight",
        "deterministic_metric_local",
    ),
    ("global_first_to_local_first", "deterministic_metric_local", "order_local_first_full"),
    (
        "full_to_compact_global_first",
        "deterministic_metric_local",
        "budget_global_first_compact",
    ),
    (
        "full_to_compact_local_first",
        "order_local_first_full",
        "budget_local_first_compact",
    ),
    ("fixed_global_to_local_raw", "fixed_global", "presence_local_raw"),
    ("fixed_global_to_full_local", "fixed_global", "deterministic_metric_local"),
)
EVENT_RESULT_SCHEMA_VERSION = "cal_dataset1_visual_config_judgement_v1"
RUN_SCHEMA_VERSION = "cal_dataset1_visual_config_judgement_run_v1"
CONTRACT_SCHEMA_VERSION = "cal_dataset1_visual_config_judgement_contract_v1"


def main() -> None:
    args = _parse_args()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    contour_evidence_root = (
        Path(args.contour_evidence_root).expanduser().resolve()
        if args.contour_evidence_root
        else None
    )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    judge_config_path = Path(args.judge_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    judge_config = _read_json(judge_config_path)
    manifests = _discover_manifests(
        evidence_root,
        metrics=set(args.metric),
        severities=set(args.severity),
    )
    jobs = _build_jobs(
        manifests=manifests,
        arms=tuple(args.arm),
        evidence_root=evidence_root,
        dataset_root=dataset_root,
        judge_config=judge_config,
        judge_config_path=judge_config_path,
        out_dir=out_dir,
        contour_evidence_root=contour_evidence_root,
    )
    plan = _plan(args, evidence_root, dataset_root, judge_config_path, judge_config, jobs)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan["counts"], indent=2))
        return

    api_key_env = str(judge_config.get("api_key_env") or "")
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"judge credential environment variable is not set: {api_key_env}"
        )

    started = time.time()
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if args.resume and _visual_result_ready(job["result_path"], job["contract"]):
            results.append(_read_json(job["result_path"]))
            print(
                f"[cached] {job['case_id']} {job['metric']}:{job['event_id']} {job['arm']}",
                flush=True,
            )
        else:
            pending.append(job)

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_judge_visual_job, job, judge_config): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    result = _failed_result(job, exc)
                    result["schema_version"] = EVENT_RESULT_SCHEMA_VERSION
                    _write_json(job["result_path"], result)
                results.append(result)
                status = "ok" if result.get("predicted_label") else "failed"
                print(
                    f"[{status}] {job['case_id']} {job['metric']}:{job['event_id']} {job['arm']}",
                    flush=True,
                )

    results.sort(key=_result_sort_key)
    summaries = _summary_rows(results)
    paired_rows, paired_summaries = _paired_rows(results)
    master_rows = [_flat_result(result) for result in results]
    mixed_candidates = _mixed_camera_policy_candidates(summaries)
    _write_tsv(out_dir / "per_event.tsv", master_rows)
    _write_tsv(out_dir / "master_event_policy_table.tsv", master_rows)
    _write_tsv(out_dir / "policy_summary.tsv", summaries)
    _write_tsv(out_dir / "paired_transitions.tsv", paired_rows)
    _write_tsv(out_dir / "paired_transition_summary.tsv", paired_summaries)
    _write_tsv(out_dir / "mixed_camera_policy_candidates.tsv", mixed_candidates)
    _write_json(
        out_dir / "summary.json",
        {
            "summary": summaries,
            "paired_transition_summary": paired_summaries,
            "mixed_camera_policy_candidates": mixed_candidates,
        },
    )
    failure_count = sum(bool(result.get("error")) for result in results)
    resolved_count = sum(
        result.get("predicted_label") in {"valid", "invalid"}
        for result in results
    )
    manifest = {
        **plan,
        "schema_version": RUN_SCHEMA_VERSION,
        "elapsed_seconds": time.time() - started,
        "result_count": len(results),
        "resolved_count": resolved_count,
        "failure_count": failure_count,
        "complete": (
            len(results) == plan["counts"]["judge_calls"]
            and resolved_count == plan["counts"]["judge_calls"]
            and failure_count == 0
        ),
        "outputs": {
            "master_event_policy_table": str(
                (out_dir / "master_event_policy_table.tsv").resolve()
            ),
            "policy_summary": str((out_dir / "policy_summary.tsv").resolve()),
            "paired_transition_summary": str(
                (out_dir / "paired_transition_summary.tsv").resolve()
            ),
            "mixed_camera_policy_candidates": str(
                (out_dir / "mixed_camera_policy_candidates.tsv").resolve()
            ),
        },
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "result_count": len(results),
                "resolved_count": resolved_count,
                "failure_count": failure_count,
                "complete": manifest["complete"],
                "results": str(out_dir),
            },
            indent=2,
        )
    )
    raise SystemExit(1 if failure_count else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        default=str(PROJECT_ROOT / "Support" / "artifacts" / "outputs" / "exp1_1"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"),
    )
    parser.add_argument(
        "--contour-evidence-root",
        default=None,
        help="Prepared contour evidence root; required for presence_local_raw_contour.",
    )
    parser.add_argument(
        "--judge-config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "models"
            / "gpt5_6_sol_litellm_local_visual_config_judge.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_1_visual_config_gpt56"
        ),
    )
    parser.add_argument("--arm", action="append", choices=ALL_ARMS, default=[])
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--severity", action="append", choices=SEVERITIES, default=[])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not args.arm:
        args.arm = list(ARMS)
    if CONTOUR_ARM in args.arm and not args.contour_evidence_root:
        parser.error(
            "--contour-evidence-root is required when presence_local_raw_contour is selected"
        )
    if not args.metric:
        args.metric = list(METRICS)
    if not args.severity:
        args.severity = list(SEVERITIES)
    if not 1 <= args.max_workers <= 8:
        parser.error("--max-workers must be between 1 and 8")
    return args


def _build_jobs(
    *,
    manifests: list[Path],
    arms: tuple[str, ...],
    evidence_root: Path,
    dataset_root: Path,
    judge_config: dict[str, Any],
    judge_config_path: Path,
    out_dir: Path,
    contour_evidence_root: Path | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    identity = _judge_identity(judge_config)
    implementation_sha256 = {
        "runner": _file_sha256(Path(__file__).resolve()),
        "frozen_evidence_runner": _file_sha256(
            PROJECT_ROOT / "scripts" / "judge_cal_dataset1_camera_evidence.py"
        ),
        "p0b": _file_sha256(
            PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "p0b.py"
        ),
        "openai_compatible": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "benchmark"
            / "visual_judge"
            / "openai_compatible.py"
        ),
        "segmentation_contour": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "benchmark"
            / "rendering"
            / "segmentation_contour.py"
        ),
    }
    for manifest_path in manifests:
        comparison = _read_json(manifest_path)
        case_id = str(comparison["case_id"])
        metric = str(comparison["metric"])
        event_id = str(comparison["event_id"])
        fixture = dataset_root / "fixtures" / case_id
        source_files = _source_files(dataset_root, evidence_root, case_id, fixture)
        source_warnings = _verify_source_hashes(comparison, source_files)
        for arm in arms:
            items, factors = _compose_visual_items(
                comparison,
                arm,
                contour_evidence_root=contour_evidence_root,
            )
            verification_root = (
                evidence_root.parent
                if contour_evidence_root is not None
                else evidence_root
            )
            evidence = _verified_evidence(items, verification_root)
            result_path = (
                out_dir
                / "events"
                / case_id
                / f"{metric}__{_safe_name(event_id)}"
                / f"{arm}.json"
            )
            contract = {
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "case_id": case_id,
                "metric": metric,
                "event_id": event_id,
                "arm": arm,
                "object_ids": [str(value) for value in comparison.get("object_ids") or []],
                "severity_class": comparison.get("severity_class"),
                "presentation": factors["presentation"],
                "local_presentation": factors["local_presentation"],
                "global_context": factors["global_context"],
                "image_order": factors["image_order"],
                "image_budget": len(items),
                "max_local_views": factors["max_local_views"],
                "comparison_manifest_sha256": _file_sha256(manifest_path),
                "source_sha256": deepcopy(comparison.get("source_sha256") or {}),
                "source_integrity_warnings": source_warnings,
                "evidence": evidence,
                "judge_config_sha256": _file_sha256(judge_config_path),
                "judge_identity": identity,
                "implementation_sha256": implementation_sha256,
                "visual_config_policy": "passthrough",
                "ground_truth_visibility": "scoring_only_not_sent_to_judge",
            }
            jobs.append(
                {
                    "case_id": case_id,
                    "metric": metric,
                    "event_id": event_id,
                    "arm": arm,
                    "object_ids": list(contract["object_ids"]),
                    "severity_class": comparison.get("severity_class"),
                    "gt_label": comparison.get("semantic_label"),
                    "gt_basis": comparison.get("gt_basis"),
                    "manifest_path": manifest_path,
                    "fixture": fixture,
                    "source_files": source_files,
                    "items": items,
                    "contract": contract,
                    "result_path": result_path,
                }
            )
    jobs.sort(key=_job_sort_key)
    return jobs


def _compose_visual_items(
    comparison: dict[str, Any],
    arm: str,
    *,
    contour_evidence_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_arms = comparison.get("arms") or {}
    global_payload = source_arms.get("fixed_global_highlight") or {}
    local_payload = source_arms.get("metric_local_highlight") or {}
    global_items = deepcopy(global_payload.get("items") or [])
    local_items = deepcopy(local_payload.get("items") or [])
    if len(global_items) != 4 or len(local_items) != 4:
        raise ValueError(
            f"expected four frozen global and local items for {comparison.get('case_id')} "
            f"{comparison.get('metric')}:{comparison.get('event_id')}"
        )
    global_raw = [item for item in global_items if _is_raw(item)]
    global_highlight = [item for item in global_items if _is_highlight(item)]
    local_raw = [item for item in local_items if _is_raw(item)]
    local_highlight = [item for item in local_items if _is_highlight(item)]
    if not all(
        len(values) == 2
        for values in (global_raw, global_highlight, local_raw, local_highlight)
    ):
        raise ValueError("frozen raw/highlight pairing is incomplete")
    local_pairs = _paired_views(local_raw, local_highlight)
    global_pairs = _paired_views(global_raw, global_highlight)
    global_top_highlight = _by_view_id(global_highlight, "global_top")
    local_full = [item for pair in local_pairs for item in pair]
    first_local_pair = local_pairs[0]
    local_contour_full: list[dict[str, Any]] | None = None
    if arm == CONTOUR_ARM:
        if contour_evidence_root is None:
            raise ValueError("contour evidence root is required for contour arm")
        contour_path = (
            contour_evidence_root
            / "cases"
            / str(comparison.get("case_id"))
            / "events"
            / (
                f"{comparison.get('metric')}__"
                f"{_safe_name(str(comparison.get('event_id')))}"
            )
            / "contour_evidence_manifest.json"
        )
        contour_manifest = _read_json(contour_path)
        if contour_manifest.get("complete") is not True:
            raise RuntimeError(f"contour evidence is incomplete: {contour_path}")
        contour_items = [
            deepcopy(item)
            for item in contour_manifest.get("contour_items", [])
            if isinstance(item, dict)
        ]
        if len(contour_items) != 2:
            raise ValueError(f"expected two contour items: {contour_path}")
        contour_pairs = _paired_views(local_raw, contour_items)
        local_contour_full = [item for pair in contour_pairs for item in pair]

    definitions: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
        "fixed_global": (
            global_raw,
            _factors("raw", "none", "two_frozen_raw_overviews", "global_first", 0),
        ),
        "fixed_global_highlight": (
            [item for pair in global_pairs for item in pair],
            _factors(
                "raw_plus_same_pose_highlight",
                "none",
                "two_frozen_raw_plus_highlight_overviews",
                "global_first",
                0,
            ),
        ),
        "presence_local_raw": (
            local_raw,
            _factors("raw", "raw", "none", "local_first", 2),
        ),
        "presence_local_raw_highlight": (
            local_full,
            _factors(
                "raw_plus_same_pose_highlight",
                "raw_plus_highlight",
                "none",
                "local_first",
                2,
            ),
        ),
        "presence_local_raw_contour": (
            local_contour_full or [],
            _factors(
                "raw_plus_same_pose_segmentation_contour",
                "raw_plus_segmentation_contour",
                "none",
                "local_first",
                2,
            ),
        ),
        "presence_global_local_raw": (
            [global_top_highlight, *local_raw],
            _factors(
                "raw_plus_global_highlight",
                "raw",
                "highlighted_global_top",
                "global_first",
                2,
            ),
        ),
        "deterministic_metric_local": (
            [global_top_highlight, *local_full],
            _factors(
                "raw_plus_same_pose_highlight",
                "raw_plus_highlight",
                "highlighted_global_top",
                "global_first",
                2,
            ),
        ),
        "order_local_first_full": (
            [*local_full, global_top_highlight],
            _factors(
                "raw_plus_same_pose_highlight",
                "raw_plus_highlight",
                "highlighted_global_top",
                "local_first",
                2,
            ),
        ),
        "budget_global_first_compact": (
            [global_top_highlight, *first_local_pair],
            _factors(
                "raw_plus_same_pose_highlight",
                "raw_plus_highlight",
                "highlighted_global_top",
                "global_first",
                1,
            ),
        ),
        "budget_local_first_compact": (
            [*first_local_pair, global_top_highlight],
            _factors(
                "raw_plus_same_pose_highlight",
                "raw_plus_highlight",
                "highlighted_global_top",
                "local_first",
                1,
            ),
        ),
    }
    if arm not in definitions:
        raise ValueError(f"unsupported VisualConfig arm: {arm}")
    items, factors = definitions[arm]
    expected_budget = {
        "fixed_global": 2,
        "fixed_global_highlight": 4,
        "presence_local_raw": 2,
        "presence_local_raw_highlight": 4,
        "presence_local_raw_contour": 4,
        "presence_global_local_raw": 3,
        "deterministic_metric_local": 5,
        "order_local_first_full": 5,
        "budget_global_first_compact": 3,
        "budget_local_first_compact": 3,
    }[arm]
    if len(items) != expected_budget:
        raise RuntimeError(f"{arm} expected {expected_budget} images, got {len(items)}")
    return deepcopy(items), factors


def _factors(
    presentation: str,
    local_presentation: str,
    global_context: str,
    image_order: str,
    max_local_views: int,
) -> dict[str, Any]:
    return {
        "presentation": presentation,
        "local_presentation": local_presentation,
        "global_context": global_context,
        "image_order": image_order,
        "max_local_views": max_local_views,
    }


def _is_raw(item: dict[str, Any]) -> bool:
    return str(item.get("role") or "") in {"metric_local_rgb", "collision_rgb"}


def _is_highlight(item: dict[str, Any]) -> bool:
    return str(item.get("role") or "") in {
        "metric_local_highlight",
        "collision_pair_overlay",
    }


def _by_view_id(items: list[dict[str, Any]], view_id: str) -> dict[str, Any]:
    matches = [item for item in items if str(item.get("view_id")) == view_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen item for view_id={view_id!r}")
    return matches[0]


def _paired_views(
    raw_items: list[dict[str, Any]],
    highlight_items: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    return [
        [raw, _by_view_id(highlight_items, str(raw.get("view_id") or ""))]
        for raw in raw_items
    ]


def _judge_visual_job(job: dict[str, Any], judge_config: dict[str, Any]) -> dict[str, Any]:
    result = _judge_one(job, judge_config)
    result["schema_version"] = EVENT_RESULT_SCHEMA_VERSION
    _write_json(job["result_path"], result)
    return result


def _visual_result_ready(path: Path, contract: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
    except Exception:
        return False
    return bool(
        result.get("schema_version") == EVENT_RESULT_SCHEMA_VERSION
        and result.get("contract_sha256") == _json_sha256(contract)
        and result.get("predicted_label") in {"valid", "invalid"}
        and not result.get("error")
        and (result.get("judgement") or {}).get("verdict")
        == result.get("predicted_label")
    )


def _plan(
    args: argparse.Namespace,
    evidence_root: Path,
    dataset_root: Path,
    judge_config_path: Path,
    judge_config: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    event_keys = {
        (job["case_id"], job["metric"], job["event_id"])
        for job in jobs
    }
    first_arm = str(args.arm[0])
    return {
        "schema_version": "cal_dataset1_visual_config_judgement_plan_v1",
        "experiment_id": "exp1_1_visual_config_gpt56",
        "experiment_type": "judgement_only_frozen_visual_config_replay",
        "evidence_root": str(evidence_root),
        "dataset_root": str(dataset_root),
        "judge_config": str(judge_config_path),
        "judge_config_sha256": _file_sha256(judge_config_path),
        "judge_identity": _judge_identity(judge_config),
        "controlled_variable": "VisualConfig_only",
        "arms": list(args.arm),
        "comparisons": [
            name
            for name, left, right in PAIR_DEFINITIONS
            if left in args.arm and right in args.arm
        ],
        "frozen": {
            "scene": True,
            "prompt": True,
            "detector_evidence": True,
            "ground_truth": True,
            "visual_evidence_bytes": True,
            "camera_candidate_policy": "local",
            "renderer_invoked": False,
            "camera_selector_invoked": False,
            "vlm_choose_topk_invoked": False,
            "ground_truth_sent_to_judge": False,
        },
        "counts": {
            "cases": len({job["case_id"] for job in jobs}),
            "events": len(event_keys),
            "judge_calls": len(jobs),
            "arms": len(set(job["arm"] for job in jobs)),
            "events_by_metric": dict(
                Counter(job["metric"] for job in jobs if job["arm"] == first_arm)
            ),
            "events_by_severity": dict(
                Counter(
                    job["severity_class"]
                    for job in jobs
                    if job["arm"] == first_arm
                )
            ),
        },
        "execution": {
            "max_workers": int(args.max_workers),
            "resume": bool(args.resume),
            "continue_on_error": bool(args.continue_on_error),
        },
        "interpretation_limit": (
            "All selected GT events are invalid must-route distortions. This run supports "
            "invalid recall/FN and paired VisualConfig transitions, but cannot estimate FP "
            "or prove that highlight is safe on clean events."
        ),
    }


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ALL_ARMS:
        arm_results = [result for result in results if result.get("arm") == arm]
        if not arm_results:
            continue
        groups: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("overall", "overall", arm_results)
        ]
        groups.extend(
            (
                "metric",
                metric,
                [item for item in arm_results if item.get("metric") == metric],
            )
            for metric in METRICS
        )
        groups.extend(
            (
                "severity",
                severity,
                [item for item in arm_results if item.get("severity_class") == severity],
            )
            for severity in SEVERITIES
        )
        for group_type, group, selected in groups:
            if not selected:
                continue
            resolved = [
                item
                for item in selected
                if item.get("predicted_label") in {"valid", "invalid"}
            ]
            detected = [
                item for item in resolved if item.get("predicted_label") == "invalid"
            ]
            rows.append(
                {
                    "arm": arm,
                    "group_type": group_type,
                    "group": group,
                    "total": len(selected),
                    "resolved": len(resolved),
                    "coverage": len(resolved) / len(selected),
                    "correct": len(detected),
                    "invalid_detected": len(detected),
                    "false_negatives": sum(
                        item.get("predicted_label") == "valid" for item in selected
                    ),
                    "invalid_recall_all": len(detected) / len(selected),
                    "invalid_recall_resolved": (
                        len(detected) / len(resolved) if resolved else None
                    ),
                    "mean_confidence": _mean(
                        [item.get("confidence") for item in resolved]
                    ),
                    "mean_latency_seconds": _mean(
                        [item.get("elapsed_seconds") for item in resolved]
                    ),
                    "mean_image_budget": _mean(
                        [item.get("image_budget") for item in selected]
                    ),
                    "error_count": sum(bool(item.get("error")) for item in selected),
                }
            )
    return rows


def _paired_rows(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        key = (
            str(result.get("case_id")),
            str(result.get("metric")),
            str(result.get("event_id")),
        )
        indexed[key][str(result.get("arm"))] = result
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison, left_arm, right_arm in PAIR_DEFINITIONS:
        pairs = [
            (key, arms[left_arm], arms[right_arm])
            for key, arms in indexed.items()
            if left_arm in arms and right_arm in arms
        ]
        for key, left, right in pairs:
            rows.append(
                {
                    "comparison": comparison,
                    "case_id": key[0],
                    "metric": key[1],
                    "event_id": key[2],
                    "severity_class": left.get("severity_class"),
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "left_prediction": left.get("predicted_label") or "unresolved",
                    "right_prediction": right.get("predicted_label") or "unresolved",
                    "left_correct": int(bool(left.get("match"))),
                    "right_correct": int(bool(right.get("match"))),
                    "transition": _transition(left, right),
                }
            )
        groups: list[
            tuple[str, str, list[tuple[tuple[str, str, str], dict[str, Any], dict[str, Any]]]]
        ] = [("overall", "overall", pairs)]
        groups.extend(
            ("metric", metric, [pair for pair in pairs if pair[0][1] == metric])
            for metric in METRICS
        )
        groups.extend(
            (
                "severity",
                severity,
                [pair for pair in pairs if pair[1].get("severity_class") == severity],
            )
            for severity in SEVERITIES
        )
        for group_type, group, selected in groups:
            if not selected:
                continue
            transitions = Counter(
                _transition(left, right) for _, left, right in selected
            )
            summaries.append(
                {
                    "comparison": comparison,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "group_type": group_type,
                    "group": group,
                    "total_pairs": len(selected),
                    "both_correct": transitions["both_correct"],
                    "left_only_correct": transitions["left_only_correct"],
                    "right_only_correct": transitions["right_only_correct"],
                    "both_incorrect": transitions["both_incorrect"],
                    "unresolved": transitions["unresolved"],
                    "right_minus_left_correct": (
                        transitions["right_only_correct"]
                        - transitions["left_only_correct"]
                    ),
                }
            )
    return rows, summaries


def _transition(left: dict[str, Any], right: dict[str, Any]) -> str:
    if not left.get("resolved") or not right.get("resolved"):
        return "unresolved"
    if left.get("match") and right.get("match"):
        return "both_correct"
    if left.get("match"):
        return "left_only_correct"
    if right.get("match"):
        return "right_only_correct"
    return "both_incorrect"


def _mixed_camera_policy_candidates(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        candidates = [
            row
            for row in summaries
            if row["group_type"] == "metric"
            and row["group"] == metric
            and row["resolved"] == row["total"]
        ]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda row: (
                -float(row["invalid_recall_all"]),
                float(row["mean_image_budget"] or 999),
                ALL_ARMS.index(str(row["arm"])),
            ),
        )[0]
        rows.append(
            {
                "metric": metric,
                "candidate_arm": best["arm"],
                "invalid_recall": best["invalid_recall_all"],
                "mean_image_budget": best["mean_image_budget"],
                "selection_rule": "maximize_invalid_recall_then_minimize_image_budget",
                "interpretation": "pilot_candidate_only_invalid_GT_cannot_measure_FP",
            }
        )
    return rows


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "metric": result.get("metric"),
        "event_id": result.get("event_id"),
        "severity_class": result.get("severity_class"),
        "arm": result.get("arm"),
        "presentation": result.get("presentation"),
        "local_presentation": result.get("local_presentation"),
        "global_context": result.get("global_context"),
        "image_order": result.get("image_order"),
        "image_budget": result.get("image_budget"),
        "max_local_views": result.get("max_local_views"),
        "gt_label": result.get("gt_label"),
        "predicted_label": result.get("predicted_label") or "unresolved",
        "resolved": int(bool(result.get("resolved"))),
        "match": int(bool(result.get("match"))),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "judge_model": (result.get("judge_identity") or {}).get("model"),
        "comparison_manifest_sha256": result.get("comparison_manifest_sha256"),
        "contract_sha256": result.get("contract_sha256"),
        "error": result.get("error"),
    }


def _job_sort_key(job: dict[str, Any]) -> tuple[Any, ...]:
    return (
        METRICS.index(str(job["metric"])),
        SEVERITIES.index(str(job["severity_class"])),
        str(job["case_id"]),
        str(job["event_id"]),
        ALL_ARMS.index(str(job["arm"])),
    )


def _result_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return _job_sort_key(result)


def _mean(values: list[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numeric) / len(numeric) if numeric else None


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
