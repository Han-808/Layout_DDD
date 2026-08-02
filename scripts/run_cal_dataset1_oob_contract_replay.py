#!/usr/bin/env python3
"""Re-run only cal_dataset1 OOB outcomes after an OOB contract change.

Historical experiment outputs are immutable.  This runner recomputes current
OOB detector evidence from canonical scenes, reuses byte-frozen RGB/highlight
evidence where requested, and writes a separate result universe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator.generic_validity.oob import (  # noqa: E402
    DEFAULT_OOB_CONFIG,
    OOB_EVALUATOR_VERSION,
    check_oob,
)
from benchmark.scene_io.object_normalization import normalize_objects  # noqa: E402
from benchmark.visual_judge.openai_compatible import (  # noqa: E402
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.p0b import adjudicate_p0b_event  # noqa: E402
from scripts.judge_cal_dataset1_visual_config import (  # noqa: E402
    ARMS as NINE_ARMS,
    _compose_visual_items,
)


EXPERIMENTS = ("dataset", "two_arm", "nine_arm")
TWO_ARMS = ("fixed_global_highlight", "metric_local_highlight")
RESULT_SCHEMA_VERSION = "cal_dataset1_oob_contract_replay_result_v1"
RUN_SCHEMA_VERSION = "cal_dataset1_oob_contract_replay_run_v1"
REQUIRED_V2_FIELDS = {
    "floor_contact_tolerance_m",
    "plane_penetration_m",
    "within_floor_contact_tolerance",
}


def main() -> None:
    args = _parse_args()
    contract = _require_current_oob_contract()
    if args.check_contract:
        print(json.dumps(contract, indent=2))
        return

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    judge_config_path = Path(args.judge_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    evidence_root = (
        Path(args.evidence_root).expanduser().resolve()
        if args.evidence_root
        else None
    )
    judge_config = _read_json(judge_config_path)
    api_key_env = str(judge_config.get("api_key_env") or "")
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"judge credential environment variable is not set: {api_key_env}"
        )

    if args.experiment == "dataset":
        jobs = _dataset_jobs(
            dataset_root=dataset_root,
            out_dir=out_dir,
            splits=set(args.split),
            gt_labels=set(args.gt_label),
            judge_config_path=judge_config_path,
            contract=contract,
        )
    else:
        if evidence_root is None:
            raise ValueError("--evidence-root is required for frozen-arm replays")
        jobs = _frozen_jobs(
            experiment=args.experiment,
            dataset_root=dataset_root,
            evidence_root=evidence_root,
            out_dir=out_dir,
            splits=set(args.split),
            gt_labels=set(args.gt_label),
            judge_config_path=judge_config_path,
            contract=contract,
        )
    if not jobs:
        raise RuntimeError("selection produced no OOB replay jobs")

    plan = {
        "schema_version": "cal_dataset1_oob_contract_replay_plan_v1",
        "experiment": args.experiment,
        "dataset_root": str(dataset_root),
        "evidence_root": str(evidence_root) if evidence_root else None,
        "out_dir": str(out_dir),
        "oob_contract": contract,
        "splits": list(args.split),
        "gt_labels": list(args.gt_label),
        "job_count": len(jobs),
        "event_count": len(
            {(job["case_id"], job["event_id"]) for job in jobs}
        ),
        "arms": sorted({str(job["arm"]) for job in jobs}),
        "renderer_invoked": False,
        "camera_selector_invoked": False,
        "historical_outputs_modified": False,
        "ground_truth_sent_to_judge": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    started = time.time()
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        cached = _cached_result(job["result_path"], job["contract"])
        if args.resume and cached is not None:
            results.append(cached)
            print(
                f"[cached] {job['case_id']} oob:{job['event_id']} {job['arm']}",
                flush=True,
            )
        else:
            pending.append(job)

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_run_job, job, judge_config): job
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
                    _write_json(job["result_path"], result)
                results.append(result)
                status = (
                    "direct"
                    if result.get("judge_called") is False
                    and result.get("predicted_label") == "valid"
                    else "ok"
                    if result.get("predicted_label") in {"valid", "invalid"}
                    else "failed"
                )
                print(
                    f"[{status}] {job['case_id']} oob:{job['event_id']} "
                    f"{job['arm']}",
                    flush=True,
                )

    results.sort(
        key=lambda result: (
            str(result.get("arm")),
            str(result.get("gt_label")),
            str(result.get("case_id")),
            str(result.get("event_id")),
        )
    )
    rows = [_flat_result(result) for result in results]
    summaries = _summary_rows(rows)
    _write_tsv(out_dir / "per_event.tsv", rows)
    _write_tsv(out_dir / "summary.tsv", summaries)
    _write_json(
        out_dir / "summary.json",
        {
            "schema_version": "cal_dataset1_oob_contract_replay_summary_v1",
            "summary": summaries,
        },
    )
    manifest = {
        **plan,
        "schema_version": RUN_SCHEMA_VERSION,
        "elapsed_seconds": time.time() - started,
        "result_count": len(results),
        "resolved_count": sum(
            result.get("predicted_label") in {"valid", "invalid"}
            for result in results
        ),
        "direct_valid_count": sum(
            result.get("judge_called") is False
            and result.get("predicted_label") == "valid"
            for result in results
        ),
        "vlm_call_count": sum(result.get("judge_called") is True for result in results),
        "failure_count": sum(bool(result.get("error")) for result in results),
        "outputs": {
            "per_event": str((out_dir / "per_event.tsv").resolve()),
            "summary": str((out_dir / "summary.tsv").resolve()),
        },
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "result_count": manifest["result_count"],
                "resolved_count": manifest["resolved_count"],
                "direct_valid_count": manifest["direct_valid_count"],
                "vlm_call_count": manifest["vlm_call_count"],
                "failure_count": manifest["failure_count"],
                "summary": manifest["outputs"]["summary"],
            },
            indent=2,
        )
    )
    if manifest["failure_count"]:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="dataset")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"),
    )
    parser.add_argument("--evidence-root", default="")
    parser.add_argument(
        "--judge-config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "models"
            / "gpt5_6_sol_litellm_local_fine_edge_judge.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp3_oob_v2_gpt56"
        ),
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument(
        "--gt-label",
        action="append",
        choices=("valid", "invalid", "ambiguous"),
        default=[],
    )
    parser.add_argument("--max-workers", type=int, default=2)
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
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check-contract", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    if not args.split:
        args.split = [
            "source_valid",
            "obvious_distortion",
            "subtle_distortion",
            "fine_edge",
        ]
    if not args.gt_label:
        args.gt_label = ["valid", "invalid", "ambiguous"]
    return args


def _require_current_oob_contract() -> dict[str, Any]:
    tolerance = DEFAULT_OOB_CONFIG.get("floor_contact_tolerance_m")
    if (
        OOB_EVALUATOR_VERSION == "oob_p0b_v1"
        or tolerance is None
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise RuntimeError(
            "The OOB floor-contact fix is not present. Expected a post-v1 "
            "evaluator and DEFAULT_OOB_CONFIG.floor_contact_tolerance_m."
        )
    return {
        "evaluator_version": OOB_EVALUATOR_VERSION,
        "numerical_eps": float(DEFAULT_OOB_CONFIG["numerical_eps"]),
        "floor_contact_tolerance_m": float(tolerance),
        "oob_implementation_sha256": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "benchmark"
            / "evaluator"
            / "generic_validity"
            / "oob.py"
        ),
        "p0b_rubric_sha256": _file_sha256(
            PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "p0b.py"
        ),
    }


def _dataset_jobs(
    *,
    dataset_root: Path,
    out_dir: Path,
    splits: set[str],
    gt_labels: set[str],
    judge_config_path: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = _read_json(dataset_root / "cases.json").get("cases") or []
    jobs: list[dict[str, Any]] = []
    for case in cases:
        if str(case.get("split")) not in splits:
            continue
        case_id = str(case["case_id"])
        fixture = dataset_root / str(case["fixture_dir"])
        scene_path = fixture / "generated_scene.json"
        request_path = fixture / "scene_request.json"
        plan_path = fixture / "object_plan.json"
        gt_path = fixture / "event_gt.json"
        gt = _read_json(gt_path)
        for event in gt.get("events") or []:
            if (
                not isinstance(event, dict)
                or event.get("metric") != "oob"
                or str(event.get("semantic_label")) not in gt_labels
            ):
                continue
            event_id = str(event["event_id"])
            overview = _dataset_overview_items(dataset_root, case_id)
            job_contract = _job_contract(
                experiment="dataset",
                case_id=case_id,
                event_id=event_id,
                arm="metric_default_global_context",
                scene_path=scene_path,
                request_path=request_path,
                plan_path=plan_path,
                gt_path=gt_path,
                manifest_path=None,
                items=overview,
                judge_config_path=judge_config_path,
                oob_contract=contract,
            )
            jobs.append(
                {
                    "experiment": "dataset",
                    "case_id": case_id,
                    "split": str(case.get("split")),
                    "event_id": event_id,
                    "object_ids": [str(value) for value in event.get("object_ids") or []],
                    "gt_label": str(event.get("semantic_label")),
                    "gt_basis": event.get("gt_basis"),
                    "severity_class": event.get("severity_class"),
                    "arm": "metric_default_global_context",
                    "fixture": fixture,
                    "scene_path": scene_path,
                    "items": overview,
                    "visual_config_policy": "metric_default",
                    "contract": job_contract,
                    "result_path": _result_path(
                        out_dir, case_id, event_id, "metric_default_global_context"
                    ),
                }
            )
    return jobs


def _frozen_jobs(
    *,
    experiment: str,
    dataset_root: Path,
    evidence_root: Path,
    out_dir: Path,
    splits: set[str],
    gt_labels: set[str],
    judge_config_path: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = TWO_ARMS if experiment == "two_arm" else tuple(NINE_ARMS)
    jobs: list[dict[str, Any]] = []
    manifests = sorted(
        evidence_root.glob("cases/*/events/oob__*/comparison_manifest.json")
    )
    for manifest_path in manifests:
        comparison = _read_json(manifest_path)
        if comparison.get("metric") != "oob":
            continue
        split = str(comparison.get("split") or "")
        gt_label = str(comparison.get("semantic_label") or "")
        if split not in splits or gt_label not in gt_labels:
            continue
        case_id = str(comparison["case_id"])
        event_id = str(comparison["event_id"])
        fixture = dataset_root / "fixtures" / case_id
        scene_path = fixture / "generated_scene.json"
        request_path = fixture / "scene_request.json"
        plan_path = fixture / "object_plan.json"
        gt_path = fixture / "event_gt.json"
        for arm in arms:
            if experiment == "two_arm":
                payload = (comparison.get("arms") or {}).get(arm)
                if not isinstance(payload, dict):
                    raise ValueError(f"manifest lacks {arm}: {manifest_path}")
                items = deepcopy(payload.get("items") or [])
            else:
                items, _factors = _compose_visual_items(comparison, arm)
            verified = _verified_items(items, evidence_root)
            job_contract = _job_contract(
                experiment=experiment,
                case_id=case_id,
                event_id=event_id,
                arm=arm,
                scene_path=scene_path,
                request_path=request_path,
                plan_path=plan_path,
                gt_path=gt_path,
                manifest_path=manifest_path,
                items=verified,
                judge_config_path=judge_config_path,
                oob_contract=contract,
            )
            jobs.append(
                {
                    "experiment": experiment,
                    "case_id": case_id,
                    "split": split,
                    "event_id": event_id,
                    "object_ids": [
                        str(value)
                        for value in comparison.get("object_ids") or []
                    ],
                    "gt_label": gt_label,
                    "gt_basis": comparison.get("gt_basis"),
                    "severity_class": comparison.get("severity_class"),
                    "arm": arm,
                    "fixture": fixture,
                    "scene_path": scene_path,
                    "items": verified,
                    "visual_config_policy": "passthrough",
                    "contract": job_contract,
                    "result_path": _result_path(out_dir, case_id, event_id, arm),
                }
            )
    return jobs


def _run_job(
    job: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    scene = _read_json(job["scene_path"])
    detector, event, record = _current_detector_context(
        scene, str(job["event_id"])
    )
    prompt, relationships = _prompt_context(Path(job["fixture"]))
    candidate = bool(record.get("candidate_oob"))
    if not candidate:
        predicted = "valid"
        judgement = None
        judge_called = False
        route = str(record.get("route") or "direct_valid")
        confidence = 1.0
        reason = (
            "Current deterministic OOB contract classifies the measured "
            "geometry as direct valid."
        )
    else:
        judge = build_openai_compatible_vlm_judge(deepcopy(judge_config))
        if job["visual_config_policy"] == "metric_default":
            overview_paths = [str(item["path"]) for item in job["items"]]
            provider = None
        else:
            overview_paths = []
            provider = lambda _request: deepcopy(job["items"])
        judgement = adjudicate_p0b_event(
            metric="oob",
            event=event,
            prompt=prompt,
            relationships=relationships,
            scene=scene,
            detector_evidence=detector,
            judge=judge,
            object_ids=list(job["object_ids"]),
            overview_render_evidence=overview_paths,
            local_view_provider=provider,
            visual_config_policy=str(job["visual_config_policy"]),
        )
        predicted = str(judgement["verdict"])
        judge_called = True
        route = "vlm_adjudicated"
        confidence = judgement.get("confidence")
        reason = judgement.get("reason")

    gt_label = str(job["gt_label"])
    scoreable = gt_label in {"valid", "invalid"}
    result = {
        **deepcopy(job["contract"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "case_id": job["case_id"],
        "split": job["split"],
        "event_id": job["event_id"],
        "arm": job["arm"],
        "gt_label": gt_label,
        "gt_basis": job["gt_basis"],
        "severity_class": job["severity_class"],
        "predicted_label": predicted,
        "resolved": predicted in {"valid", "invalid"},
        "match": predicted == gt_label if scoreable else None,
        "route": route,
        "requires_vlm": candidate,
        "judge_called": judge_called,
        "confidence": confidence,
        "reason": reason,
        "plane_flags": deepcopy(record.get("plane_flags") or {}),
        "plane_penetration_m": deepcopy(
            record.get("plane_penetration_m") or {}
        ),
        "within_floor_contact_tolerance": bool(
            record.get("within_floor_contact_tolerance")
        ),
        "floor_contact_tolerance_m": detector["floor_contact_tolerance_m"],
        "numerical_eps": detector["numerical_eps"],
        "judgement": judgement,
        "elapsed_seconds": time.time() - started,
        "error": None,
    }
    _write_json(job["result_path"], result)
    return result


def _current_detector_context(
    scene: dict[str, Any],
    object_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = check_oob(
        scene,
        {
            "official_mode": False,
            "detector_only": True,
            "numerical_eps": DEFAULT_OOB_CONFIG["numerical_eps"],
            "floor_contact_tolerance_m": DEFAULT_OOB_CONFIG[
                "floor_contact_tolerance_m"
            ],
        },
    )
    record = next(
        (
            item
            for item in report.get("objects") or []
            if str(item.get("object_id")) == object_id
        ),
        None,
    )
    if not isinstance(record, dict):
        raise RuntimeError(f"current OOB report lacks object {object_id}")
    missing = sorted(REQUIRED_V2_FIELDS - set(record))
    if missing:
        raise RuntimeError(
            f"current OOB record lacks required v2 fields for {object_id}: {missing}"
        )
    normalized, _errors = normalize_objects(scene)
    obj = next((value for value in normalized if str(value.id) == object_id), None)
    if obj is None:
        raise RuntimeError(f"normalized scene lacks object {object_id}")
    detector = {
        "detector": OOB_EVALUATOR_VERSION,
        "plane_flags": deepcopy(record.get("plane_flags") or {}),
        "obb_intervals": deepcopy(record.get("obb_intervals") or {}),
        "plane_penetration_m": deepcopy(record["plane_penetration_m"]),
        "room": deepcopy(report.get("room") or {}),
        "numerical_eps": float(DEFAULT_OOB_CONFIG["numerical_eps"]),
        "floor_contact_tolerance_m": float(
            DEFAULT_OOB_CONFIG["floor_contact_tolerance_m"]
        ),
        "within_floor_contact_tolerance": bool(
            record["within_floor_contact_tolerance"]
        ),
        "object": {
            "id": str(obj.id),
            "category": obj.category,
            "description": obj.desc,
            "center": [float(value) for value in obj.center],
            "size": [float(value) for value in obj.size],
            "rotation_degrees": [float(value) for value in obj.rotation],
            "geometry_provenance": _geometry_provenance(scene, object_id),
        },
        "extracted_relationships_are_claims_only": True,
    }
    for key in (
        "semantic_tolerances_m",
        "semantic_thresholds_m",
        "raw_plane_crossings",
    ):
        if key in record:
            detector[key] = deepcopy(record[key])
    event = {
        "object_id": object_id,
        "object_ids": [object_id],
        "architecture_element": "room_bounds",
        "plane_flags": deepcopy(record.get("plane_flags") or {}),
    }
    return detector, event, record


def _dataset_overview_items(
    dataset_root: Path,
    case_id: str,
) -> list[dict[str, Any]]:
    render_dir = (
        dataset_root / "evaluation" / "mesh_geometry" / case_id / "renders"
    )
    candidates = (
        ("global_top", render_dir / "standardized_top.png"),
        ("global_perspective", render_dir / "standardized_perspective.png"),
    )
    items = []
    for view_id, path in candidates:
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append(
            {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
                "role": "overview_rgb",
                "view_id": view_id,
            }
        )
    return items


def _verified_items(
    items: list[dict[str, Any]],
    evidence_root: Path,
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("frozen evidence item lacks path")
        path = Path(str(item["path"])).expanduser().resolve()
        try:
            path.relative_to(evidence_root)
        except ValueError as exc:
            raise RuntimeError(f"evidence escapes root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _file_sha256(path)
        if digest != item.get("sha256"):
            raise RuntimeError(f"frozen evidence hash mismatch: {path}")
        verified.append(
            {
                **deepcopy(item),
                "path": str(path),
                "sha256": digest,
            }
        )
    return verified


def _job_contract(
    *,
    experiment: str,
    case_id: str,
    event_id: str,
    arm: str,
    scene_path: Path,
    request_path: Path,
    plan_path: Path,
    gt_path: Path,
    manifest_path: Path | None,
    items: list[dict[str, Any]],
    judge_config_path: Path,
    oob_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "case_id": case_id,
        "metric": "oob",
        "event_id": event_id,
        "arm": arm,
        "scene_sha256": _file_sha256(scene_path),
        "scene_request_sha256": _file_sha256(request_path),
        "object_plan_sha256": _file_sha256(plan_path),
        "gt_sha256": _file_sha256(gt_path),
        "comparison_manifest_sha256": (
            _file_sha256(manifest_path) if manifest_path else None
        ),
        "judge_config_sha256": _file_sha256(judge_config_path),
        "oob_contract": deepcopy(oob_contract),
        "evidence": [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "role": item.get("role"),
                "view_id": item.get("view_id"),
            }
            for item in items
        ],
        "renderer_invoked": False,
        "camera_selector_invoked": False,
        "ground_truth_sent_to_judge": False,
    }


def _prompt_context(fixture: Path) -> tuple[str, list[dict[str, Any]]]:
    request = _read_json(fixture / "scene_request.json")
    plan = _read_json(fixture / "object_plan.json")
    return (
        str(request.get("instruction") or ""),
        list(plan.get("relations") or []),
    )


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", rows)
    ]
    groups.extend(
        (
            "arm",
            arm,
            [row for row in rows if row["arm"] == arm],
        )
        for arm in sorted({str(row["arm"]) for row in rows})
    )
    groups.extend(
        (
            "gt_label",
            label,
            [row for row in rows if row["gt_label"] == label],
        )
        for label in ("valid", "invalid", "ambiguous")
        if any(row["gt_label"] == label for row in rows)
    )
    summaries: list[dict[str, Any]] = []
    for group_type, group, selected in groups:
        resolved = [
            row
            for row in selected
            if row["predicted_label"] in {"valid", "invalid"}
        ]
        scoreable = [
            row for row in selected if row["gt_label"] in {"valid", "invalid"}
        ]
        correct = [row for row in scoreable if row["match"] == 1]
        valid_gt = [row for row in selected if row["gt_label"] == "valid"]
        invalid_gt = [row for row in selected if row["gt_label"] == "invalid"]
        summaries.append(
            {
                "group_type": group_type,
                "group": group,
                "total": len(selected),
                "resolved": len(resolved),
                "scoreable": len(scoreable),
                "correct": len(correct),
                "accuracy": (
                    len(correct) / len(scoreable) if scoreable else None
                ),
                "valid_gt": len(valid_gt),
                "false_positives": sum(
                    row["predicted_label"] == "invalid" for row in valid_gt
                ),
                "specificity": (
                    sum(row["predicted_label"] == "valid" for row in valid_gt)
                    / len(valid_gt)
                    if valid_gt
                    else None
                ),
                "invalid_gt": len(invalid_gt),
                "true_positives": sum(
                    row["predicted_label"] == "invalid" for row in invalid_gt
                ),
                "invalid_recall": (
                    sum(
                        row["predicted_label"] == "invalid"
                        for row in invalid_gt
                    )
                    / len(invalid_gt)
                    if invalid_gt
                    else None
                ),
                "predicted_invalid": sum(
                    row["predicted_label"] == "invalid" for row in selected
                ),
                "direct_valid": sum(
                    row["judge_called"] == 0
                    and row["predicted_label"] == "valid"
                    for row in selected
                ),
                "vlm_calls": sum(row["judge_called"] == 1 for row in selected),
                "errors": sum(bool(row["error"]) for row in selected),
            }
        )
    return summaries


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "split": result.get("split"),
        "metric": "oob",
        "event_id": result.get("event_id"),
        "arm": result.get("arm"),
        "gt_label": result.get("gt_label"),
        "predicted_label": result.get("predicted_label"),
        "resolved": int(bool(result.get("resolved"))),
        "match": (
            ""
            if result.get("match") is None
            else int(bool(result.get("match")))
        ),
        "route": result.get("route"),
        "requires_vlm": int(bool(result.get("requires_vlm"))),
        "judge_called": int(bool(result.get("judge_called"))),
        "within_floor_contact_tolerance": int(
            bool(result.get("within_floor_contact_tolerance"))
        ),
        "floor_contact_tolerance_m": result.get(
            "floor_contact_tolerance_m"
        ),
        "numerical_eps": result.get("numerical_eps"),
        "plane_flags": json.dumps(
            result.get("plane_flags") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "plane_penetration_m": json.dumps(
            result.get("plane_penetration_m") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "error": result.get("error"),
    }


def _cached_result(
    path: Path,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = _read_json(path)
    except Exception:
        return None
    if (
        result.get("schema_version") == RESULT_SCHEMA_VERSION
        and result.get("contract_sha256") == _json_sha256(contract)
        and result.get("predicted_label") in {"valid", "invalid"}
        and not result.get("error")
    ):
        return result
    return None


def _failed_result(
    job: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        **deepcopy(job["contract"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "case_id": job["case_id"],
        "split": job["split"],
        "event_id": job["event_id"],
        "arm": job["arm"],
        "gt_label": job["gt_label"],
        "gt_basis": job["gt_basis"],
        "severity_class": job["severity_class"],
        "predicted_label": None,
        "resolved": False,
        "match": None,
        "route": "failed",
        "requires_vlm": None,
        "judge_called": None,
        "confidence": None,
        "reason": None,
        "plane_flags": {},
        "plane_penetration_m": {},
        "within_floor_contact_tolerance": None,
        "floor_contact_tolerance_m": DEFAULT_OOB_CONFIG.get(
            "floor_contact_tolerance_m"
        ),
        "numerical_eps": DEFAULT_OOB_CONFIG.get("numerical_eps"),
        "judgement": None,
        "elapsed_seconds": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _result_path(
    out_dir: Path,
    case_id: str,
    event_id: str,
    arm: str,
) -> Path:
    return (
        out_dir
        / "events"
        / _safe_name(case_id)
        / f"oob__{_safe_name(event_id)}"
        / f"{_safe_name(arm)}.json"
    )


def _geometry_provenance(scene: dict[str, Any], object_id: str) -> Any:
    for item in scene.get("objects") or []:
        if isinstance(item, dict) and str(item.get("id")) == object_id:
            return item.get("geometry_provenance")
    return None


def _safe_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in str(value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
