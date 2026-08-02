#!/usr/bin/env python3
"""Judge the frozen nested Support Top-K evidence ablation.

Ground truth is applied only after each model call.  The judge receives the
same event context in every arm; only the number of deterministic local
highlight views changes (K=1/2/3), with one identical global context image
appended last.
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
from itertools import combinations
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
    _event_context,
    _prompt_context,
    _report_record,
)


INPUT_SCHEMA = "cal_dataset1_support_topk_comparison_v1"
RESULT_SCHEMA = "cal_dataset1_support_topk_judgement_v1"
CONTRACT_SCHEMA = "cal_dataset1_support_topk_judgement_contract_v1"
RUN_SCHEMA = "cal_dataset1_support_topk_judgement_run_v1"
ARMS = ("support_top1", "support_top2", "support_top3")


def main() -> None:
    args = _parse_args()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    judge_config_path = Path(args.judge_config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    judge_config = _read_json(judge_config_path)
    manifests = _discover_manifests(evidence_root)
    jobs = _build_jobs(
        manifests=manifests,
        arms=tuple(args.arm),
        dataset_root=dataset_root,
        judge_config=judge_config,
        judge_config_path=judge_config_path,
        out_dir=out_dir,
    )
    plan = _plan(
        args,
        evidence_root,
        dataset_root,
        judge_config_path,
        judge_config,
        jobs,
    )
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
            results.append(_read_json(job["result_path"]))
            print(
                f"[cached] {job['case_id']} support:{job['event_id']} {job['arm']}",
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
                    f"[{status}] {job['case_id']} support:{job['event_id']} {job['arm']}",
                    flush=True,
                )

    results.sort(key=_result_sort_key)
    summaries = _summary_rows(results)
    paired = _paired_rows(results, tuple(args.arm))
    _write_tsv(out_dir / "per_event.tsv", [_flat_result(item) for item in results])
    _write_tsv(out_dir / "summary.tsv", summaries)
    _write_tsv(out_dir / "paired_transition_summary.tsv", paired)
    _write_json(
        out_dir / "summary.json",
        {"summary": summaries, "paired_transitions": paired},
    )
    run = {
        **plan,
        "schema_version": RUN_SCHEMA,
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
    _write_json(out_dir / "run_manifest.json", run)
    print(json.dumps({
        "result_count": run["result_count"],
        "resolved_count": run["resolved_count"],
        "failure_count": run["failure_count"],
        "summary": run["outputs"]["summary"],
        "paired_transitions": run["outputs"]["paired_transitions"],
    }, indent=2))
    if run["failure_count"]:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_support_topk_evidence"
        ),
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
            / "exp2_support_topk_gpt56"
        ),
    )
    parser.add_argument("--arm", action="append", choices=ARMS, default=[])
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
    if not 1 <= args.max_workers <= 8:
        parser.error("--max-workers must be between 1 and 8")
    return args


def _discover_manifests(evidence_root: Path) -> list[Path]:
    if not evidence_root.is_dir():
        raise FileNotFoundError(evidence_root)
    prepare_run_path = evidence_root / "run_manifest.json"
    if not prepare_run_path.is_file():
        raise FileNotFoundError(
            f"Support TopK preparation run manifest is missing: {prepare_run_path}"
        )
    prepare_run = _read_json(prepare_run_path)
    planned_events = int((prepare_run.get("counts") or {}).get("events") or 0)
    completed_events = int(prepare_run.get("completed_event_count") or 0)
    failed_events = int(prepare_run.get("failed_event_count") or 0)
    if (
        planned_events <= 0
        or completed_events != planned_events
        or failed_events != 0
    ):
        raise RuntimeError(
            "Support TopK evidence is incomplete; refusing a partial ablation: "
            f"planned={planned_events}, completed={completed_events}, failed={failed_events}"
        )
    paths = sorted(
        evidence_root.glob("cases/*/events/support__*/comparison_manifest.json")
    )
    if len(paths) != planned_events:
        raise RuntimeError(
            "Support TopK manifest count does not match preparation plan: "
            f"planned={planned_events}, found={len(paths)}"
        )
    if not paths:
        raise ValueError("no prepared Support TopK manifests found")
    seen: set[tuple[str, str]] = set()
    for path in paths:
        value = _read_json(path)
        if value.get("schema_version") != INPUT_SCHEMA:
            raise ValueError(f"unsupported Support TopK schema: {path}")
        key = (str(value.get("case_id")), str(value.get("event_id")))
        if key in seen:
            raise ValueError(f"duplicate Support TopK event: {key}")
        seen.add(key)
    return [path.resolve() for path in paths]


def _build_jobs(
    *,
    manifests: list[Path],
    arms: tuple[str, ...],
    dataset_root: Path,
    judge_config: dict[str, Any],
    judge_config_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    max_images = int(judge_config.get("max_images") or 0)
    judge_identity = _judge_identity(judge_config)
    implementation_sha256 = {
        "runner": _file_sha256(Path(__file__).resolve()),
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
    }
    jobs: list[dict[str, Any]] = []
    for manifest_path in manifests:
        comparison = _read_json(manifest_path)
        case_id = str(comparison["case_id"])
        event_id = str(comparison["event_id"])
        fixture = dataset_root / "fixtures" / case_id
        source_files = {
            "scene": fixture / "generated_scene.json",
            "event_gt": fixture / "event_gt.json",
            "scene_request": fixture / "scene_request.json",
            "object_plan": fixture / "object_plan.json",
            "detector_report": (
                dataset_root
                / "evaluation"
                / "mesh"
                / case_id
                / "generic_validity.json"
            ),
        }
        _verify_dataset_sources(comparison, source_files)
        for arm in arms:
            arm_payload = (comparison.get("arms") or {}).get(arm)
            if not isinstance(arm_payload, dict):
                raise ValueError(f"manifest lacks arm {arm}: {manifest_path}")
            items = deepcopy(arm_payload.get("items") or [])
            expected_budget = int(arm_payload.get("image_count") or 0)
            if len(items) != expected_budget:
                raise ValueError(
                    f"image budget mismatch for {case_id} support:{event_id} {arm}"
                )
            if max_images and expected_budget > max_images:
                raise ValueError(
                    f"judge max_images={max_images} cannot accept {arm} budget={expected_budget}"
                )
            evidence = _verified_evidence(items)
            contract = {
                "schema_version": CONTRACT_SCHEMA,
                "case_id": case_id,
                "metric": "support",
                "event_id": event_id,
                "arm": arm,
                "object_ids": [
                    str(value) for value in comparison.get("object_ids") or []
                ],
                "severity_class": comparison.get("severity_class"),
                "presentation": comparison.get("presentation"),
                "image_order": comparison.get("image_order"),
                "image_budget": len(evidence),
                "local_view_count": arm_payload.get("local_view_count"),
                "comparison_manifest_sha256": _file_sha256(manifest_path),
                "source_contract": deepcopy(comparison.get("source_contract") or {}),
                "source_sha256": deepcopy(comparison.get("source_sha256") or {}),
                "evidence": evidence,
                "judge_config_sha256": _file_sha256(judge_config_path),
                "judge_identity": judge_identity,
                "implementation_sha256": implementation_sha256,
                "visual_config_policy": "passthrough",
                "ground_truth_visibility": "scoring_only_not_sent_to_judge",
            }
            result_path = (
                out_dir
                / "events"
                / case_id
                / f"support__{_safe_name(event_id)}"
                / f"{arm}.json"
            )
            jobs.append({
                "case_id": case_id,
                "event_id": event_id,
                "metric": "support",
                "arm": arm,
                "arm_index": ARMS.index(arm),
                "object_ids": list(contract["object_ids"]),
                "severity_class": comparison.get("severity_class"),
                "gt_label": comparison.get("semantic_label"),
                "gt_basis": comparison.get("gt_basis"),
                "fixture": fixture,
                "source_files": source_files,
                "items": items,
                "contract": contract,
                "result_path": result_path,
            })
    jobs.sort(key=lambda job: (
        str(job["severity_class"]),
        str(job["case_id"]),
        str(job["event_id"]),
        int(job["arm_index"]),
    ))
    return jobs


def _verify_dataset_sources(
    comparison: dict[str, Any],
    source_files: dict[str, Path],
) -> None:
    expected = comparison.get("source_sha256")
    if not isinstance(expected, dict):
        raise ValueError("TopK manifest lacks source_sha256")
    for key, path in source_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = expected.get(key)
        if expected_hash is not None and _file_sha256(path) != expected_hash:
            raise RuntimeError(f"dataset source hash mismatch for {key}: {path}")


def _verified_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError("each evidence item requires path and sha256")
        path = Path(str(item["path"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {path}")
        verified.append({
            "index": index,
            "path": str(path),
            "sha256": actual,
            "role": item.get("role"),
            "view_id": item.get("view_id"),
        })
    return verified


def _judge_one(
    job: dict[str, Any],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    scene = _read_json(job["source_files"]["scene"])
    report = _read_json(job["source_files"]["detector_report"])
    prompt, relationships = _prompt_context(Path(job["fixture"]))
    record = _report_record(
        report,
        "support",
        str(job["event_id"]),
        list(job["object_ids"]),
    )
    detector_evidence, event_payload = _event_context(
        "support",
        record,
        {"object_ids": list(job["object_ids"])},
    )
    judge = build_openai_compatible_vlm_judge(deepcopy(judge_config))
    judgement = adjudicate_p0b_event(
        metric="support",
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
        "schema_version": RESULT_SCHEMA,
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
        "schema_version": RESULT_SCHEMA,
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
            result.get("schema_version") == RESULT_SCHEMA
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
    first_arm = args.arm[0]
    events = [job for job in jobs if job["arm"] == first_arm]
    return {
        "schema_version": "cal_dataset1_support_topk_judgement_plan_v1",
        "experiment_id": "exp2_support_topk_gpt56",
        "experiment_type": "judgement_only_nested_support_topk_ablation",
        "evidence_root": str(evidence_root),
        "dataset_root": str(dataset_root),
        "judge_config": str(judge_config_path),
        "judge_config_sha256": _file_sha256(judge_config_path),
        "judge_identity": _judge_identity(judge_config),
        "controlled_variable": "number_of_deterministically_ranked_local_highlight_views",
        "arms": list(args.arm),
        "frozen": {
            "scene": True,
            "prompt": True,
            "detector_evidence": True,
            "ground_truth": True,
            "candidate_bank": True,
            "selector": "support_contact_plane_visibility_rank_v1",
            "presentation": "highlight_only",
            "global_context": "same_existing_metric_highlighted_global",
            "image_order": "local_first_then_global",
            "renderer_invoked": False,
            "camera_selector_invoked": False,
            "ground_truth_sent_to_judge": False,
        },
        "counts": {
            "events": len(events),
            "judge_calls": len(jobs),
            "arms": len(args.arm),
            "events_by_gt_label": dict(
                Counter(str(job["gt_label"]) for job in events)
            ),
            "events_by_severity": dict(
                Counter(str(job["severity_class"]) for job in events)
            ),
        },
        "execution": {
            "max_workers": int(args.max_workers),
            "resume": bool(args.resume),
            "continue_on_error": bool(args.continue_on_error),
        },
        "interpretation_limit": (
            "Invalid events estimate Support recall/FN. Ambiguous events have no "
            "accuracy label and are reported only as invalid/valid tendency. "
            "This experiment does not estimate false-positive rate."
        ),
    }


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_results = [item for item in results if item.get("arm") == arm]
        if not arm_results:
            continue
        groups: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("overall", "overall", arm_results)
        ]
        groups.extend(
            (
                "severity",
                severity,
                [
                    item for item in arm_results
                    if str(item.get("severity_class")) == severity
                ],
            )
            for severity in ("obvious", "subtle", "edge")
        )
        groups.extend([
            (
                "gt_label",
                label,
                [
                    item for item in arm_results
                    if str(item.get("gt_label")) == label
                ],
            )
            for label in ("invalid", "ambiguous")
        ])
        for group_type, group, selected in groups:
            if not selected:
                continue
            resolved = [
                item for item in selected
                if item.get("predicted_label") in {"valid", "invalid"}
            ]
            invalid_gt = [
                item for item in selected if item.get("gt_label") == "invalid"
            ]
            ambiguous = [
                item for item in selected if item.get("gt_label") == "ambiguous"
            ]
            invalid_detected = [
                item for item in invalid_gt
                if item.get("predicted_label") == "invalid"
            ]
            ambiguous_resolved = [
                item for item in ambiguous
                if item.get("predicted_label") in {"valid", "invalid"}
            ]
            rows.append({
                "arm": arm,
                "local_view_count": int(arm[-1]),
                "image_budget": int(arm[-1]) + 1,
                "group_type": group_type,
                "group": group,
                "total": len(selected),
                "resolved": len(resolved),
                "coverage": len(resolved) / len(selected),
                "invalid_gt_total": len(invalid_gt),
                "invalid_detected": len(invalid_detected),
                "false_negatives": sum(
                    item.get("predicted_label") == "valid"
                    for item in invalid_gt
                ),
                "invalid_recall_all": (
                    len(invalid_detected) / len(invalid_gt)
                    if invalid_gt else None
                ),
                "ambiguous_total": len(ambiguous),
                "ambiguous_resolved": len(ambiguous_resolved),
                "ambiguous_predicted_invalid": sum(
                    item.get("predicted_label") == "invalid"
                    for item in ambiguous_resolved
                ),
                "ambiguous_invalid_rate": (
                    sum(
                        item.get("predicted_label") == "invalid"
                        for item in ambiguous_resolved
                    ) / len(ambiguous_resolved)
                    if ambiguous_resolved else None
                ),
                "mean_confidence": _mean(
                    [item.get("confidence") for item in resolved]
                ),
                "mean_latency_seconds": _mean(
                    [item.get("elapsed_seconds") for item in resolved]
                ),
                "error_count": sum(bool(item.get("error")) for item in selected),
            })
    return rows


def _paired_rows(
    results: list[dict[str, Any]],
    selected_arms: tuple[str, ...],
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        key = (str(result.get("case_id")), str(result.get("event_id")))
        indexed[key][str(result.get("arm"))] = result
    rows: list[dict[str, Any]] = []
    for lower, higher in combinations(selected_arms, 2):
        pairs = [
            pair for pair in indexed.values()
            if lower in pair and higher in pair
        ]
        groups: list[tuple[str, str, list[dict[str, dict[str, Any]]]]] = [
            ("overall", "overall", pairs)
        ]
        groups.extend(
            (
                "gt_label",
                label,
                [
                    pair for pair in pairs
                    if pair[lower].get("gt_label") == label
                ],
            )
            for label in ("invalid", "ambiguous")
        )
        for group_type, group, selected in groups:
            if not selected:
                continue
            resolved = [
                pair for pair in selected
                if pair[lower].get("predicted_label") in {"valid", "invalid"}
                and pair[higher].get("predicted_label") in {"valid", "invalid"}
            ]
            rows.append({
                "lower_arm": lower,
                "higher_arm": higher,
                "group_type": group_type,
                "group": group,
                "total_pairs": len(selected),
                "resolved_pairs": len(resolved),
                "verdict_agreement": (
                    sum(
                        pair[lower]["predicted_label"]
                        == pair[higher]["predicted_label"]
                        for pair in resolved
                    ) / len(resolved)
                    if resolved else None
                ),
                "both_invalid": sum(
                    pair[lower]["predicted_label"] == "invalid"
                    and pair[higher]["predicted_label"] == "invalid"
                    for pair in resolved
                ),
                "both_valid": sum(
                    pair[lower]["predicted_label"] == "valid"
                    and pair[higher]["predicted_label"] == "valid"
                    for pair in resolved
                ),
                "lower_invalid_higher_valid": sum(
                    pair[lower]["predicted_label"] == "invalid"
                    and pair[higher]["predicted_label"] == "valid"
                    for pair in resolved
                ),
                "lower_valid_higher_invalid": sum(
                    pair[lower]["predicted_label"] == "valid"
                    and pair[higher]["predicted_label"] == "invalid"
                    for pair in resolved
                ),
            })
    return rows


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": result.get("case_id"),
        "event_id": result.get("event_id"),
        "severity_class": result.get("severity_class"),
        "arm": result.get("arm"),
        "local_view_count": result.get("local_view_count"),
        "image_budget": result.get("image_budget"),
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
        str(result.get("severity_class")),
        str(result.get("case_id")),
        str(result.get("event_id")),
        ARMS.index(str(result.get("arm"))),
    )


def _mean(values: list[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numeric) / len(numeric) if numeric else None


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._") or "event"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
