#!/usr/bin/env python3
"""Judge frozen cal_dataset1 camera evidence without rendering or selection.

The runner consumes the two byte-frozen arms produced by
``run_cal_dataset1_camera_evidence.py``.  It reconstructs the P0b event context
from the frozen dataset, calls the existing binary P0b VLM judge, and reports
per-arm recall or ambiguous-event tendency plus paired camera-policy
transitions. GT labels are used only after the model call for scoring and are
never included in the judge request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from benchmark.visual_judge.openai_compatible import (  # noqa: E402
    build_openai_compatible_vlm_judge,
)
from benchmark.visual_judge.p0b import adjudicate_p0b_event  # noqa: E402
from scripts.run_cal_dataset1_camera_evidence import (  # noqa: E402
    COMPARISON_SCHEMA_VERSION,
    _event_context,
    _prompt_context,
    _report_record,
)


ARMS = ("fixed_global_highlight", "metric_local_highlight")
METRICS = ("collision", "oob", "support")
SEVERITIES = ("obvious", "subtle", "edge")
EVENT_RESULT_SCHEMA_VERSION = "cal_dataset1_frozen_evidence_judgement_v1"
RUN_SCHEMA_VERSION = "cal_dataset1_frozen_evidence_judgement_run_v1"
CONTRACT_SCHEMA_VERSION = "cal_dataset1_frozen_evidence_judgement_contract_v1"


def main() -> None:
    args = _parse_args()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    judge_config_path = Path(args.judge_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    judge_config = _read_json(judge_config_path)
    manifests = _discover_manifests(
        evidence_root,
        metrics=set(args.metric),
        severities=set(args.severity),
        case_ids=set(args.case_id),
    )
    jobs = _build_jobs(
        manifests=manifests,
        arms=tuple(args.arm),
        evidence_root=evidence_root,
        dataset_root=dataset_root,
        judge_config=judge_config,
        judge_config_path=judge_config_path,
        out_dir=out_dir,
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
        if args.resume and _result_ready(job["result_path"], job["contract"]):
            result = _read_json(job["result_path"])
            results.append(result)
            print(
                f"[cached] {job['case_id']} {job['metric']}:{job['event_id']} {job['arm']}",
                flush=True,
            )
        else:
            pending.append(job)

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_judge_one, job, judge_config): job
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
                status = "ok" if result.get("predicted_label") else "failed"
                print(
                    f"[{status}] {job['case_id']} {job['metric']}:{job['event_id']} {job['arm']}",
                    flush=True,
                )

    results.sort(key=_result_sort_key)
    summaries = _summary_rows(results)
    paired = _paired_rows(results)
    _write_tsv(out_dir / "per_event.tsv", [_flat_result(result) for result in results])
    _write_tsv(out_dir / "summary.tsv", summaries)
    _write_tsv(out_dir / "paired_transition_summary.tsv", paired)
    _write_json(
        out_dir / "summary.json",
        {"summary": summaries, "paired_transitions": paired},
    )
    run_manifest = {
        **plan,
        "schema_version": RUN_SCHEMA_VERSION,
        "elapsed_seconds": time.time() - started,
        "result_count": len(results),
        "resolved_count": sum(
            result.get("predicted_label") in {"valid", "invalid"}
            for result in results
        ),
        "failure_count": sum(bool(result.get("error")) for result in results),
        "outputs": {
            "per_event": str((out_dir / "per_event.tsv").resolve()),
            "summary": str((out_dir / "summary.tsv").resolve()),
            "paired_transitions": str(
                (out_dir / "paired_transition_summary.tsv").resolve()
            ),
        },
    }
    _write_json(out_dir / "run_manifest.json", run_manifest)
    print(json.dumps({
        "result_count": run_manifest["result_count"],
        "resolved_count": run_manifest["resolved_count"],
        "failure_count": run_manifest["failure_count"],
        "summary": str((out_dir / "summary.tsv").resolve()),
        "paired_transitions": str((out_dir / "paired_transition_summary.tsv").resolve()),
    }, indent=2))


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
        "--judge-config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "models"
            / "gpt5_6_sol_litellm_local_judge.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_1_gpt56_judge"
        ),
    )
    parser.add_argument("--arm", action="append", choices=ARMS, default=[])
    parser.add_argument("--metric", action="append", choices=METRICS, default=[])
    parser.add_argument("--severity", action="append", choices=SEVERITIES, default=[])
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Restrict judgement to one or more exact case IDs.",
    )
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
    if not args.metric:
        args.metric = list(METRICS)
    if not args.severity:
        args.severity = list(SEVERITIES)
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8")
    return args


def _discover_manifests(
    evidence_root: Path,
    *,
    metrics: set[str],
    severities: set[str],
    case_ids: set[str] | None = None,
) -> list[Path]:
    if not evidence_root.is_dir():
        raise FileNotFoundError(evidence_root)
    case_ids = case_ids or set()
    paths: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    found_case_ids: set[str] = set()
    for path in sorted(evidence_root.glob("cases/*/events/*/comparison_manifest.json")):
        comparison = _read_json(path)
        if comparison.get("schema_version") != COMPARISON_SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison schema: {path}")
        metric = str(comparison.get("metric") or "")
        severity = str(comparison.get("severity_class") or "")
        case_id = str(comparison.get("case_id") or "")
        if metric not in metrics or severity not in severities:
            continue
        if case_ids and case_id not in case_ids:
            continue
        found_case_ids.add(case_id)
        key = (
            case_id,
            metric,
            str(comparison.get("event_id") or ""),
        )
        if key in seen:
            raise ValueError(f"duplicate frozen event manifest: {key}")
        seen.add(key)
        paths.append(path.resolve())
    if not paths:
        raise ValueError("no frozen comparison manifests match the selected filters")
    missing_case_ids = case_ids - found_case_ids
    if missing_case_ids:
        raise ValueError(
            "requested case IDs have no matching frozen manifests: "
            + ", ".join(sorted(missing_case_ids))
        )
    return paths


def _build_jobs(
    *,
    manifests: list[Path],
    arms: tuple[str, ...],
    evidence_root: Path,
    dataset_root: Path,
    judge_config: dict[str, Any],
    judge_config_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    judge_identity = _judge_identity(judge_config)
    implementation_sha256 = {
        "runner": _file_sha256(Path(__file__).resolve()),
        "p0b": _file_sha256(PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "p0b.py"),
        "openai_compatible": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "benchmark"
            / "visual_judge"
            / "openai_compatible.py"
        ),
    }
    for manifest_path in manifests:
        comparison = _read_json(manifest_path)
        case_id = str(comparison["case_id"])
        metric = str(comparison["metric"])
        event_id = str(comparison["event_id"])
        fixture = dataset_root / "fixtures" / case_id
        if not fixture.is_dir():
            raise FileNotFoundError(f"fixture does not exist: {fixture}")
        source_files = _source_files(dataset_root, evidence_root, case_id, fixture)
        source_integrity_warnings = _verify_source_hashes(comparison, source_files)
        for arm in arms:
            arm_payload = (comparison.get("arms") or {}).get(arm)
            if not isinstance(arm_payload, dict):
                raise ValueError(f"comparison manifest lacks arm {arm}: {manifest_path}")
            items = deepcopy(arm_payload.get("items") or [])
            if len(items) != int(comparison.get("image_budget_per_arm") or 0):
                raise ValueError(
                    f"frozen image budget mismatch for {case_id} {metric}:{event_id} {arm}"
                )
            evidence = _verified_evidence(items, evidence_root)
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
                "presentation": comparison.get("presentation"),
                "image_budget": len(evidence),
                "comparison_manifest_sha256": _file_sha256(manifest_path),
                "source_sha256": deepcopy(comparison.get("source_sha256") or {}),
                "source_integrity_warnings": source_integrity_warnings,
                "evidence": evidence,
                "judge_config_sha256": _file_sha256(judge_config_path),
                "judge_identity": judge_identity,
                "implementation_sha256": implementation_sha256,
                "visual_config_policy": "passthrough",
                "ground_truth_visibility": "scoring_only_not_sent_to_judge",
            }
            jobs.append({
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
            })
    jobs.sort(key=lambda job: (
        str(job["metric"]),
        str(job["severity_class"]),
        str(job["case_id"]),
        str(job["event_id"]),
        ARMS.index(str(job["arm"])),
    ))
    return jobs


def _source_files(
    dataset_root: Path,
    evidence_root: Path,
    case_id: str,
    fixture: Path,
) -> dict[str, Path]:
    return {
        "scene": fixture / "generated_scene.json",
        "event_gt": fixture / "event_gt.json",
        "scene_request": fixture / "scene_request.json",
        "object_plan": fixture / "object_plan.json",
        "detector_report": (
            dataset_root / "evaluation" / "mesh" / case_id / "generic_validity.json"
        ),
        "collision_geometry_manifest": (
            dataset_root
            / "evaluation"
            / "mesh_geometry"
            / case_id
            / "renders"
            / "collision_geometry_manifest.json"
        ),
        "scene_materialization": (
            evidence_root
            / "cases"
            / case_id
            / "scene"
            / "materialization_provenance.json"
        ),
    }


def _verify_source_hashes(
    comparison: dict[str, Any],
    source_files: dict[str, Path],
) -> list[str]:
    expected = comparison.get("source_sha256")
    if not isinstance(expected, dict):
        raise ValueError("comparison manifest lacks source_sha256")
    warnings: list[str] = []
    for key, path in source_files.items():
        expected_hash = expected.get(key)
        if expected_hash is None and key == "collision_geometry_manifest":
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != expected_hash:
            if key == "scene_materialization":
                _validate_materialization_provenance(
                    path,
                    expected_scene_sha256=str(expected.get("scene") or ""),
                )
                warnings.append(
                    "scene_materialization_provenance_file_was_rewritten_after_evidence_freeze; "
                    "current provenance is internally hash-consistent and frozen image hashes remain exact"
                )
                continue
            raise RuntimeError(f"frozen source hash mismatch for {key}: {path}")
    return warnings


def _validate_materialization_provenance(
    path: Path,
    *,
    expected_scene_sha256: str,
) -> None:
    provenance = _read_json(path)
    base = provenance.get("base") if isinstance(provenance.get("base"), dict) else {}
    if base.get("scene_sha256") != expected_scene_sha256:
        raise RuntimeError(f"materialization provenance scene hash mismatch: {path}")
    blend_file = path.parent / "scene.blend"
    if (
        not blend_file.is_file()
        or _file_sha256(blend_file) != provenance.get("blend_file_sha256")
    ):
        raise RuntimeError(f"materialization provenance blend mismatch: {blend_file}")
    # The renderer may append diagnostic information to render_manifest.json
    # after materialization.  It is not a judge input; the immutable .blend,
    # canonical scene, detector packet, and every selected image remain strict.
    for item in provenance.get("asset_files") or []:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise RuntimeError(f"invalid materialization asset provenance: {path}")
        asset_path = Path(str(item["path"])).expanduser()
        if not asset_path.is_file() or _file_sha256(asset_path) != item["sha256"]:
            raise RuntimeError(f"materialization asset hash mismatch: {asset_path}")


def _verified_evidence(items: list[dict[str, Any]], evidence_root: Path) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("each frozen evidence item requires path and sha256")
        path = Path(str(item["path"])).expanduser().resolve()
        try:
            path.relative_to(evidence_root)
        except ValueError as exc:
            raise RuntimeError(f"frozen evidence escapes evidence root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"frozen evidence hash mismatch: {path}")
        verified.append({
            "index": index,
            "path": str(path),
            "sha256": actual,
            "role": item.get("role"),
            "view_id": item.get("view_id"),
        })
    return verified


def _judge_one(job: dict[str, Any], judge_config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    fixture = Path(job["fixture"])
    scene = _read_json(job["source_files"]["scene"])
    report = _read_json(job["source_files"]["detector_report"])
    prompt, relationships = _prompt_context(fixture)
    record = _report_record(
        report,
        str(job["metric"]),
        str(job["event_id"]),
        list(job["object_ids"]),
    )
    # Only identity fields are passed to this helper.  The semantic label and
    # severity remain outside the model request and are applied after judging.
    detector_evidence, event_payload = _event_context(
        str(job["metric"]),
        record,
        {"object_ids": list(job["object_ids"])},
    )
    judge = build_openai_compatible_vlm_judge(deepcopy(judge_config))
    judgement = adjudicate_p0b_event(
        metric=str(job["metric"]),
        event=event_payload,
        prompt=prompt,
        relationships=relationships,
        scene=scene,
        detector_evidence=detector_evidence,
        judge=judge,
        object_ids=list(job["object_ids"]),
        overview_render_evidence=[],
        local_view_provider=lambda _request: deepcopy(job["items"]),
        visual_config_policy="passthrough",
    )
    predicted = judgement["verdict"]
    gt_label = str(job["gt_label"])
    scoreable = gt_label in {"valid", "invalid"}
    result = {
        **deepcopy(job["contract"]),
        "schema_version": EVENT_RESULT_SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "gt_label": job["gt_label"],
        "gt_basis": job["gt_basis"],
        "predicted_label": predicted,
        "resolved": True,
        "match": predicted == gt_label if scoreable else None,
        "confidence": judgement.get("confidence"),
        "reason": judgement.get("reason"),
        "elapsed_seconds": time.time() - started,
        "judgement": judgement,
        "error": None,
    }
    _write_json(job["result_path"], result)
    return result


def _failed_result(job: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **deepcopy(job["contract"]),
        "schema_version": EVENT_RESULT_SCHEMA_VERSION,
        "contract_sha256": _json_sha256(job["contract"]),
        "gt_label": job["gt_label"],
        "gt_basis": job["gt_basis"],
        "predicted_label": None,
        "resolved": False,
        "match": None,
        "confidence": None,
        "reason": None,
        "elapsed_seconds": None,
        "judgement": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _result_ready(path: Path, contract: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        return bool(
            result.get("schema_version") == EVENT_RESULT_SCHEMA_VERSION
            and result.get("contract_sha256") == _json_sha256(contract)
            and result.get("predicted_label") in {"valid", "invalid"}
            and not result.get("error")
            and result.get("judgement", {}).get("verdict")
            == result.get("predicted_label")
        )
    except Exception:
        return False


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
    first_arm = args.arm[0]
    event_jobs = [job for job in jobs if job["arm"] == first_arm]
    gt_label_counts = dict(Counter(str(job["gt_label"]) for job in event_jobs))
    if set(gt_label_counts) == {"invalid"}:
        interpretation_limit = (
            "All selected GT events are invalid must-route distortions; this run measures "
            "invalid recall/FN and paired arm transitions, not false-positive rate."
        )
    else:
        interpretation_limit = (
            "Binary correctness and invalid recall exclude GT=ambiguous events. "
            "Ambiguous events are reported through verdict tendency, arm agreement, "
            "confidence, and exact-input repeatability."
        )
    return {
        "schema_version": "cal_dataset1_frozen_evidence_judgement_plan_v1",
        "experiment_id": "exp1_1_gpt56_judge",
        "experiment_type": "judgement_only_frozen_visual_evidence_replay",
        "evidence_root": str(evidence_root),
        "dataset_root": str(dataset_root),
        "judge_config": str(judge_config_path),
        "judge_config_sha256": _file_sha256(judge_config_path),
        "judge_identity": _judge_identity(judge_config),
        "controlled_variable": "camera_pose_policy",
        "arms": list(args.arm),
        "frozen": {
            "scene": True,
            "prompt": True,
            "detector_evidence": True,
            "ground_truth": True,
            "visual_evidence_bytes": True,
            "presentation": "raw_plus_same_pose_highlight",
            "image_budget_per_arm": 4,
            "camera_candidate_policy": "local",
            "renderer_invoked": False,
            "camera_selector_invoked": False,
            "ground_truth_sent_to_judge": False,
        },
        "counts": {
            "cases": len({job["case_id"] for job in jobs}),
            "events": len(event_keys),
            "judge_calls": len(jobs),
            "arms": len(set(job["arm"] for job in jobs)),
            "events_by_metric": dict(Counter(job["metric"] for job in jobs if job["arm"] == args.arm[0])),
            "events_by_severity": dict(Counter(job["severity_class"] for job in jobs if job["arm"] == args.arm[0])),
            "events_by_gt_label": gt_label_counts,
        },
        "execution": {
            "max_workers": int(args.max_workers),
            "resume": bool(args.resume),
            "continue_on_error": bool(args.continue_on_error),
        },
        "interpretation_limit": interpretation_limit,
    }


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_results = [result for result in results if result.get("arm") == arm]
        if not arm_results:
            continue
        groups: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("overall", "overall", arm_results)
        ]
        groups.extend(
            ("metric", metric, [result for result in arm_results if result.get("metric") == metric])
            for metric in METRICS
        )
        groups.extend(
            ("severity", severity, [result for result in arm_results if result.get("severity_class") == severity])
            for severity in SEVERITIES
        )
        for group_type, group, selected in groups:
            if not selected:
                continue
            resolved = [
                result
                for result in selected
                if result.get("predicted_label") in {"valid", "invalid"}
            ]
            scored = [
                result
                for result in selected
                if result.get("gt_label") in {"valid", "invalid"}
            ]
            invalid_gt = [
                result for result in selected if result.get("gt_label") == "invalid"
            ]
            invalid_detected = [
                result
                for result in invalid_gt
                if result.get("predicted_label") == "invalid"
            ]
            ambiguous = [
                result for result in selected if result.get("gt_label") == "ambiguous"
            ]
            ambiguous_resolved = [
                result
                for result in ambiguous
                if result.get("predicted_label") in {"valid", "invalid"}
            ]
            ambiguous_invalid = [
                result
                for result in ambiguous_resolved
                if result.get("predicted_label") == "invalid"
            ]
            rows.append({
                "arm": arm,
                "group_type": group_type,
                "group": group,
                "total": len(selected),
                "resolved": len(resolved),
                "coverage": len(resolved) / len(selected),
                "scored_total": len(scored),
                "correct": sum(result.get("match") is True for result in scored),
                "accuracy_scored": (
                    sum(result.get("match") is True for result in scored) / len(scored)
                    if scored
                    else None
                ),
                "invalid_gt_total": len(invalid_gt),
                "invalid_detected": len(invalid_detected),
                "false_negatives": sum(
                    result.get("predicted_label") == "valid" for result in invalid_gt
                ),
                "invalid_recall_all": (
                    len(invalid_detected) / len(invalid_gt) if invalid_gt else None
                ),
                "invalid_recall_resolved": (
                    len(invalid_detected)
                    / sum(
                        result.get("predicted_label") in {"valid", "invalid"}
                        for result in invalid_gt
                    )
                    if any(
                        result.get("predicted_label") in {"valid", "invalid"}
                        for result in invalid_gt
                    )
                    else None
                ),
                "ambiguous_total": len(ambiguous),
                "ambiguous_resolved": len(ambiguous_resolved),
                "ambiguous_predicted_invalid": len(ambiguous_invalid),
                "ambiguous_predicted_valid": (
                    len(ambiguous_resolved) - len(ambiguous_invalid)
                ),
                "ambiguous_invalid_rate": (
                    len(ambiguous_invalid) / len(ambiguous_resolved)
                    if ambiguous_resolved
                    else None
                ),
                "mean_confidence": _mean(
                    [result.get("confidence") for result in resolved]
                ),
                "mean_latency_seconds": _mean(
                    [result.get("elapsed_seconds") for result in resolved]
                ),
                "error_count": sum(bool(result.get("error")) for result in selected),
            })
    return rows


def _paired_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        key = (
            str(result.get("case_id")),
            str(result.get("metric")),
            str(result.get("event_id")),
        )
        indexed[key][str(result.get("arm"))] = result
    pairs = [pair for pair in indexed.values() if all(arm in pair for arm in ARMS)]
    groups: list[tuple[str, str, list[dict[str, dict[str, Any]]]]] = [
        ("overall", "overall", pairs)
    ]
    groups.extend(
        ("metric", metric, [pair for pair in pairs if pair[ARMS[0]].get("metric") == metric])
        for metric in METRICS
    )
    groups.extend(
        (
            "severity",
            severity,
            [pair for pair in pairs if pair[ARMS[0]].get("severity_class") == severity],
        )
        for severity in SEVERITIES
    )
    rows: list[dict[str, Any]] = []
    for group_type, group, selected in groups:
        if not selected:
            continue
        fixed_only = 0
        local_only = 0
        both_correct = 0
        both_incorrect = 0
        unresolved = 0
        scored_pairs = 0
        ambiguous_pairs = 0
        ambiguous_resolved_pairs = 0
        ambiguous_both_invalid = 0
        ambiguous_both_valid = 0
        ambiguous_fixed_invalid_local_valid = 0
        ambiguous_fixed_valid_local_invalid = 0
        for pair in selected:
            fixed = pair[ARMS[0]]
            local = pair[ARMS[1]]
            is_ambiguous = fixed.get("gt_label") == "ambiguous"
            if is_ambiguous:
                ambiguous_pairs += 1
            if not fixed.get("resolved") or not local.get("resolved"):
                unresolved += 1
            elif is_ambiguous:
                ambiguous_resolved_pairs += 1
                fixed_prediction = fixed.get("predicted_label")
                local_prediction = local.get("predicted_label")
                if fixed_prediction == "invalid" and local_prediction == "invalid":
                    ambiguous_both_invalid += 1
                elif fixed_prediction == "valid" and local_prediction == "valid":
                    ambiguous_both_valid += 1
                elif fixed_prediction == "invalid":
                    ambiguous_fixed_invalid_local_valid += 1
                else:
                    ambiguous_fixed_valid_local_invalid += 1
            elif fixed.get("match") and local.get("match"):
                scored_pairs += 1
                both_correct += 1
            elif fixed.get("match"):
                scored_pairs += 1
                fixed_only += 1
            elif local.get("match"):
                scored_pairs += 1
                local_only += 1
            else:
                scored_pairs += 1
                both_incorrect += 1
        rows.append({
            "group_type": group_type,
            "group": group,
            "total_pairs": len(selected),
            "scored_pairs": scored_pairs,
            "ambiguous_pairs": ambiguous_pairs,
            "ambiguous_resolved_pairs": ambiguous_resolved_pairs,
            "both_correct": both_correct,
            "fixed_only_correct": fixed_only,
            "local_only_correct": local_only,
            "both_incorrect": both_incorrect,
            "unresolved_pairs": unresolved,
            "local_minus_fixed_correct": local_only - fixed_only,
            "ambiguous_both_invalid": ambiguous_both_invalid,
            "ambiguous_both_valid": ambiguous_both_valid,
            "ambiguous_fixed_invalid_local_valid": (
                ambiguous_fixed_invalid_local_valid
            ),
            "ambiguous_fixed_valid_local_invalid": (
                ambiguous_fixed_valid_local_invalid
            ),
            "ambiguous_verdict_agreement": (
                (ambiguous_both_invalid + ambiguous_both_valid)
                / ambiguous_resolved_pairs
                if ambiguous_resolved_pairs
                else None
            ),
        })
    return rows


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "metric": result.get("metric"),
        "event_id": result.get("event_id"),
        "severity_class": result.get("severity_class"),
        "arm": result.get("arm"),
        "gt_label": result.get("gt_label"),
        "predicted_label": result.get("predicted_label") or "unresolved",
        "resolved": int(bool(result.get("resolved"))),
        "match": (
            ""
            if result.get("match") is None
            else int(bool(result.get("match")))
        ),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "image_budget": result.get("image_budget"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "judge_model": (result.get("judge_identity") or {}).get("model"),
        "comparison_manifest_sha256": result.get("comparison_manifest_sha256"),
        "contract_sha256": result.get("contract_sha256"),
        "error": result.get("error"),
    }


def _judge_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("name"),
        "provider": config.get("provider"),
        "endpoint": config.get("endpoint"),
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "max_images": config.get("max_images"),
        "response_format_json": config.get("response_format_json"),
        "api_key_env": config.get("api_key_env"),
    }


def _result_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(result.get("metric")),
        str(result.get("severity_class")),
        str(result.get("case_id")),
        str(result.get("event_id")),
        ARMS.index(str(result.get("arm"))),
    )


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(numeric) / len(numeric) if numeric else None


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("._") or "event"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
