#!/usr/bin/env python3
"""Aggregate a four-policy P0b visual-evidence replay experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_ARMS = (
    "fixed_global",
    "deterministic_metric_local",
    "vlm_select_from_candidates",
    "active_metric_local",
)
VISUAL_CONFIG_ARMS = (
    "presence_local_raw",
    "presence_local_raw_highlight",
    "presence_global_local_raw",
    "order_local_first_full",
    "budget_global_first_compact",
    "budget_local_first_compact",
)
SUPPORTED_ARMS = tuple(dict.fromkeys((*DEFAULT_ARMS, *VISUAL_CONFIG_ARMS)))
METRICS = ("collision", "oob", "support")
GT_STATUS = {
    "collision": "usable_routed_event_gt",
    "oob": "usable_recall_focused_missing_broad_negative_universe",
    "support": "provisional_until_rendered_mesh_gt_contract_is_frozen",
}
PAIR_DEFINITIONS = (
    ("fixed_to_deterministic", "fixed_global", "deterministic_metric_local"),
    (
        "deterministic_to_vlm_select",
        "deterministic_metric_local",
        "vlm_select_from_candidates",
    ),
    (
        "vlm_select_to_active",
        "vlm_select_from_candidates",
        "active_metric_local",
    ),
    (
        "deterministic_to_active",
        "deterministic_metric_local",
        "active_metric_local",
    ),
    ("fixed_to_vlm_select", "fixed_global", "vlm_select_from_candidates"),
    ("fixed_to_active", "fixed_global", "active_metric_local"),
    ("highlight_without_global", "presence_local_raw", "presence_local_raw_highlight"),
    ("global_with_raw", "presence_local_raw", "presence_global_local_raw"),
    (
        "global_with_raw_highlight",
        "presence_local_raw_highlight",
        "deterministic_metric_local",
    ),
    (
        "highlight_with_global",
        "presence_global_local_raw",
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        choices=SUPPORTED_ARMS,
        default=[],
        help="Arm to aggregate; repeat as needed. Defaults to all four policies.",
    )
    args = parser.parse_args()
    arms = tuple(dict.fromkeys(args.arm or DEFAULT_ARMS))
    pairs = tuple(
        item
        for item in PAIR_DEFINITIONS
        if item[1] in arms and item[2] in arms
    )

    cases = _read_tsv(args.case_manifest)
    if not cases:
        raise ValueError("case manifest contains no selected cases")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    universe = _event_universe(cases, args.gt_root)
    observed = _observed_rows(cases, args.ablation_root)
    rows = _aligned_rows(universe, observed, arms=arms)
    _assert_frozen_packet_consistency(rows)

    summaries = _summary_rows(rows, arms=arms)
    paired_rows = _paired_rows(rows, pairs=pairs)
    paired_summaries = _paired_summary_rows(paired_rows, pairs=pairs)
    mixed_camera_candidates = _mixed_camera_policy_candidates(rows, arms=arms)

    _write_tsv(args.out_dir / "master_event_policy_table.tsv", rows)
    _write_tsv(args.out_dir / "policy_summary.tsv", summaries)
    _write_tsv(args.out_dir / "paired_transitions.tsv", paired_rows)
    _write_tsv(args.out_dir / "paired_transition_summary.tsv", paired_summaries)
    if mixed_camera_candidates:
        _write_tsv(
            args.out_dir / "mixed_camera_policy_candidates.tsv",
            mixed_camera_candidates,
        )

    payload = {
        "schema_version": "p0b_visual_evidence_policy_results_v1",
        "controlled_variable": "visual_evidence_policy",
        "arms": list(arms),
        "metrics": list(METRICS),
        "gt_status": GT_STATUS,
        "case_count": len(cases),
        "physical_event_count": len(universe),
        "policy_packet_count": len(rows),
        "summary": summaries,
        "paired_transition_summary": paired_summaries,
        "mixed_camera_policy_candidates": mixed_camera_candidates,
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"cases: {len(cases)}")
    print(f"physical events: {len(universe)}")
    print(f"policy packets: {len(rows)}")
    print(f"results: {args.out_dir}")


def _event_universe(cases: list[dict[str, str]], gt_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        metric = case["metric"]
        gt_path = gt_root / f"{case['case_id']}.json"
        gt = _read_json(gt_path)
        selected = [
            event
            for event in gt.get("events") or []
            if isinstance(event, dict) and str(event.get("metric")) == metric
        ]
        if not selected:
            raise ValueError(f"no {metric} GT events in {gt_path}")
        for event in selected:
            key = (case["case_id"], metric, str(event["event_id"]))
            if key in seen:
                raise ValueError(f"duplicate physical event: {key}")
            seen.add(key)
            rows.append(
                {
                    **case,
                    "event_id": key[2],
                    "gt_label": str(event["label"]),
                    "gt_reason_code": event.get("reason_code", ""),
                    "gt_status": GT_STATUS[metric],
                }
            )
    return rows


def _observed_rows(
    cases: list[dict[str, str]],
    ablation_root: Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for case in cases:
        path = ablation_root / case["case_id"] / "per_event.tsv"
        if not path.is_file():
            continue
        for row in _read_tsv(path):
            key = (
                case["case_id"],
                str(row.get("metric") or ""),
                str(row.get("event_id") or ""),
                str(row.get("arm") or row.get("mode") or ""),
            )
            if key in result:
                raise ValueError(f"duplicate observed policy packet: {key}")
            result[key] = row
    return result


def _aligned_rows(
    universe: list[dict[str, Any]],
    observed: dict[tuple[str, str, str, str], dict[str, str]],
    *,
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in universe:
        for arm in arms:
            key = (event["case_id"], event["metric"], event["event_id"], arm)
            raw = observed.get(key)
            if raw is None:
                raw = {
                    "predicted_label": "missing",
                    "match": "0",
                    "resolved": "0",
                    "error": "missing_policy_result",
                }
            predicted = str(raw.get("predicted_label") or "missing")
            resolved = predicted in {"valid", "invalid"} and _bool(raw.get("resolved"), True)
            match = resolved and predicted == event["gt_label"]
            rows.append(
                {
                    "physical_event_id": "::".join(key[:3]),
                    **event,
                    "arm": arm,
                    "camera_mode": raw.get("camera_mode", ""),
                    "resolved_camera_mode": raw.get("resolved_camera_mode", ""),
                    "evidence_style": raw.get("evidence_style", ""),
                    "local_presentation": raw.get("local_presentation", ""),
                    "global_context": raw.get("global_context", ""),
                    "image_order": raw.get("image_order", ""),
                    "final_image_budget": raw.get("final_image_budget", ""),
                    "max_local_views": raw.get("max_local_views", ""),
                    "predicted_label": predicted,
                    "resolved": int(resolved),
                    "match": int(match),
                    "confidence": raw.get("confidence", ""),
                    "image_count": raw.get("image_count", ""),
                    "camera_evidence_seconds": raw.get("camera_evidence_seconds", ""),
                    "candidate_preview_seconds": raw.get("candidate_preview_seconds", ""),
                    "selector_seconds": raw.get("selector_seconds", ""),
                    "final_render_seconds": raw.get("final_render_seconds", ""),
                    "judge_seconds": raw.get("judge_seconds", ""),
                    "elapsed_seconds": raw.get("elapsed_seconds", ""),
                    "estimated_uncached_seconds": raw.get("estimated_uncached_seconds", ""),
                    "camera_max_steps": raw.get("camera_max_steps", ""),
                    "pose_selector_enabled": raw.get("pose_selector_enabled", ""),
                    "frozen_event_packet_sha256": raw.get("frozen_event_packet_sha256", ""),
                    "frozen_scene_sha256": raw.get("frozen_scene_sha256", ""),
                    "frozen_source_report_sha256": raw.get("frozen_source_report_sha256", ""),
                    "frozen_gt_sha256": raw.get("frozen_gt_sha256", ""),
                    "final_judge_model": raw.get("final_judge_model", ""),
                    "pose_selector_model": raw.get("pose_selector_model", ""),
                    "error": raw.get("error", ""),
                }
            )
    return rows


def _assert_frozen_packet_consistency(rows: list[dict[str, Any]]) -> None:
    hash_fields = (
        "frozen_event_packet_sha256",
        "frozen_scene_sha256",
        "frozen_source_report_sha256",
        "frozen_gt_sha256",
    )
    hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing_hashes: list[tuple[str, str, str]] = []
    gt_labels: dict[str, set[str]] = defaultdict(set)
    final_judges: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        event_id = str(row["physical_event_id"])
        missing_result = str(row.get("error") or "") == "missing_policy_result"
        for field in hash_fields:
            value = str(row.get(field) or "").strip()
            if value:
                hashes[(event_id, field)].add(value)
            elif not missing_result:
                missing_hashes.append((event_id, str(row["arm"]), field))
        gt_labels[event_id].add(str(row["gt_label"]))
        final_judge = str(row.get("final_judge_model") or "").strip()
        if final_judge:
            final_judges[event_id].add(final_judge)
        elif not missing_result:
            missing_hashes.append((event_id, str(row["arm"]), "final_judge_model"))
        if row["arm"] in {"vlm_select_from_candidates", "active_metric_local"} and not missing_result:
            selector = str(row.get("pose_selector_model") or "").strip()
            if not selector:
                missing_hashes.append((event_id, str(row["arm"]), "pose_selector_model"))
            elif final_judge and selector != final_judge:
                raise ValueError(
                    f"pose selector/final judge model drift for {event_id}: {selector} != {final_judge}"
                )
    inconsistent_hashes = {key: value for key, value in hashes.items() if len(value) > 1}
    inconsistent_gt = {key: value for key, value in gt_labels.items() if len(value) > 1}
    inconsistent_judges = {key: value for key, value in final_judges.items() if len(value) > 1}
    if missing_hashes:
        raise ValueError(f"controlled replay metadata missing: {missing_hashes}")
    if inconsistent_hashes:
        raise ValueError(f"frozen event packet drift detected: {inconsistent_hashes}")
    if inconsistent_gt:
        raise ValueError(f"GT drift detected: {inconsistent_gt}")
    if inconsistent_judges:
        raise ValueError(f"final judge model drift detected: {inconsistent_judges}")


def _summary_rows(
    rows: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metric in (*METRICS, "overall"):
        metric_rows = rows if metric == "overall" else [row for row in rows if row["metric"] == metric]
        for arm in arms:
            selected = [row for row in metric_rows if row["arm"] == arm]
            total = len(selected)
            resolved = sum(int(row["resolved"]) for row in selected)
            correct = sum(int(row["match"]) for row in selected)
            labels = Counter(str(row["gt_label"]) for row in selected)
            result.append(
                {
                    "metric": metric,
                    "gt_status": GT_STATUS.get(metric, "mixed_metric_diagnostic_only"),
                    "label_scope": _label_scope(labels),
                    "arm": arm,
                    "total": total,
                    "valid_gt": labels["valid"],
                    "invalid_gt": labels["invalid"],
                    "resolved": resolved,
                    "unresolved": total - resolved,
                    "correct": correct,
                    "deployment_accuracy": correct / total if total else None,
                    "resolved_accuracy": correct / resolved if resolved else None,
                    "coverage": resolved / total if total else None,
                    "tp": sum(row["gt_label"] == "invalid" and row["predicted_label"] == "invalid" for row in selected),
                    "fp": sum(row["gt_label"] == "valid" and row["predicted_label"] == "invalid" for row in selected),
                    "fn": sum(row["gt_label"] == "invalid" and row["predicted_label"] == "valid" for row in selected),
                    "tn": sum(row["gt_label"] == "valid" and row["predicted_label"] == "valid" for row in selected),
                    "error_events": sum(bool(str(row.get("error") or "").strip()) for row in selected),
                    "mean_images": _mean(selected, "image_count"),
                    "mean_camera_seconds": _mean(selected, "camera_evidence_seconds"),
                    "mean_judge_seconds": _mean(selected, "judge_seconds"),
                    "mean_elapsed_seconds": _mean(selected, "elapsed_seconds"),
                    "total_elapsed_seconds": _sum(selected, "elapsed_seconds"),
                }
            )
    return result


def _paired_rows(
    rows: list[dict[str, Any]],
    *,
    pairs: tuple[tuple[str, str, str], ...] = PAIR_DEFINITIONS,
) -> list[dict[str, Any]]:
    by_event_arm = {
        (str(row["physical_event_id"]), str(row["arm"])): row
        for row in rows
    }
    event_ids = sorted({str(row["physical_event_id"]) for row in rows})
    result: list[dict[str, Any]] = []
    for pair_name, baseline_arm, treatment_arm in pairs:
        for event_id in event_ids:
            baseline = by_event_arm[(event_id, baseline_arm)]
            treatment = by_event_arm[(event_id, treatment_arm)]
            result.append(
                {
                    "comparison": pair_name,
                    "physical_event_id": event_id,
                    "case_id": baseline["case_id"],
                    "family": baseline["family"],
                    "metric": baseline["metric"],
                    "gt_label": baseline["gt_label"],
                    "gt_status": baseline["gt_status"],
                    "baseline_arm": baseline_arm,
                    "treatment_arm": treatment_arm,
                    "baseline_prediction": baseline["predicted_label"],
                    "treatment_prediction": treatment["predicted_label"],
                    "baseline_resolved": baseline["resolved"],
                    "treatment_resolved": treatment["resolved"],
                    "baseline_match": baseline["match"],
                    "treatment_match": treatment["match"],
                    "transition": _transition(baseline, treatment),
                    "elapsed_seconds_delta": _number(treatment.get("elapsed_seconds"))
                    - _number(baseline.get("elapsed_seconds")),
                    "image_count_delta": _number(treatment.get("image_count"))
                    - _number(baseline.get("image_count")),
                }
            )
    return result


def _paired_summary_rows(
    rows: list[dict[str, Any]],
    *,
    pairs: tuple[tuple[str, str, str], ...] = PAIR_DEFINITIONS,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for comparison, _, _ in pairs:
        for metric in (*METRICS, "overall"):
            selected = [
                row
                for row in rows
                if row["comparison"] == comparison
                and (metric == "overall" or row["metric"] == metric)
            ]
            transitions = Counter(str(row["transition"]) for row in selected)
            result.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "gt_status": GT_STATUS.get(metric, "mixed_metric_diagnostic_only"),
                    "event_count": len(selected),
                    "wrong_to_correct": transitions["wrong_to_correct"],
                    "correct_to_wrong": transitions["correct_to_wrong"],
                    "unresolved_to_resolved": transitions["unresolved_to_resolved"],
                    "resolved_to_unresolved": transitions["resolved_to_unresolved"],
                    "correct_to_correct": transitions["correct_to_correct"],
                    "wrong_to_wrong": transitions["wrong_to_wrong"],
                    "unresolved_to_unresolved": transitions["unresolved_to_unresolved"],
                    "mean_elapsed_seconds_delta": _mean(selected, "elapsed_seconds_delta"),
                    "mean_image_count_delta": _mean(selected, "image_count_delta"),
                }
            )
    return result


def _mixed_camera_policy_candidates(
    rows: list[dict[str, Any]],
    *,
    arms: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Recombine independent metric rows without rerunning any model calls."""

    camera_arms = (
        "fixed_global",
        "deterministic_metric_local",
        "vlm_select_from_candidates",
    )
    if not set(camera_arms).issubset(arms):
        return []
    result: list[dict[str, Any]] = []
    for choices in product(camera_arms, repeat=len(METRICS)):
        policy = dict(zip(METRICS, choices, strict=True))
        selected = [row for row in rows if row["arm"] == policy.get(str(row["metric"]))]
        total = len(selected)
        resolved = sum(int(row["resolved"]) for row in selected)
        correct = sum(int(row["match"]) for row in selected)
        result.append(
            {
                "policy_id": "__".join(
                    f"{metric}={policy[metric]}" for metric in METRICS
                ),
                **{f"{metric}_arm": policy[metric] for metric in METRICS},
                "total": total,
                "resolved": resolved,
                "correct": correct,
                "deployment_accuracy": correct / total if total else None,
                "resolved_accuracy": correct / resolved if resolved else None,
                "coverage": resolved / total if total else None,
                "fp": sum(
                    row["gt_label"] == "valid" and row["predicted_label"] == "invalid"
                    for row in selected
                ),
                "fn": sum(
                    row["gt_label"] == "invalid" and row["predicted_label"] == "valid"
                    for row in selected
                ),
                "mean_images": _mean(selected, "image_count"),
                "mean_camera_seconds": _mean(selected, "camera_evidence_seconds"),
                "mean_judge_seconds": _mean(selected, "judge_seconds"),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            -_number(row.get("deployment_accuracy")),
            -_number(row.get("coverage")),
            _number(row.get("mean_images")),
            _number(row.get("mean_camera_seconds")),
        ),
    )


def _transition(baseline: dict[str, Any], treatment: dict[str, Any]) -> str:
    baseline_resolved = bool(int(baseline["resolved"]))
    treatment_resolved = bool(int(treatment["resolved"]))
    if not baseline_resolved and not treatment_resolved:
        return "unresolved_to_unresolved"
    if not baseline_resolved:
        return "unresolved_to_resolved"
    if not treatment_resolved:
        return "resolved_to_unresolved"
    baseline_match = bool(int(baseline["match"]))
    treatment_match = bool(int(treatment["match"]))
    if baseline_match and treatment_match:
        return "correct_to_correct"
    if baseline_match:
        return "correct_to_wrong"
    if treatment_match:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _label_scope(labels: Counter[str]) -> str:
    if labels["valid"] and labels["invalid"]:
        return "mixed"
    if labels["invalid"]:
        return "invalid_only_recall_focused"
    if labels["valid"]:
        return "valid_only_specificity_focused"
    return "empty"


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    return _sum(rows, field) / len(rows) if rows else None


def _sum(rows: list[dict[str, Any]], field: str) -> float:
    return sum(_number(row.get(field)) for row in rows)


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any, default: bool = False) -> bool:
    if value in {True, 1, "1", "true", "True"}:
        return True
    if value in {False, 0, "0", "false", "False"}:
        return False
    return default


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
