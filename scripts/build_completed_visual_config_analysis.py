#!/usr/bin/env python3
"""Build the completed local VisualConfig calibration analysis bundle.

This joins:

* Exp1.1 invalid recall and repeatability;
* the nine-arm VisualConfig replay;
* source-valid false-positive calibration;
* the nested Support Top-K ablation.

Ambiguous events are preserved for human review and never enter accuracy.
Large rendered evidence remains in its original output roots and is referenced
by relative path from the generated review pages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "experiment_analysis"
    / "exp1_1_visual_config_20260723"
)


def main() -> None:
    args = _parse_args()
    analysis_root = Path(args.analysis_root).expanduser().resolve()
    valid_root = Path(args.valid_root).expanduser().resolve()
    topk_root = Path(args.topk_root).expanduser().resolve()
    topk_evidence_root = Path(args.topk_evidence_root).expanduser().resolve()
    extended_root = Path(args.extended_root).expanduser().resolve()
    nine_arm_root = Path(args.nine_arm_root).expanduser().resolve()
    oob_v2_root = Path(args.oob_v2_root).expanduser().resolve()

    required = [
        valid_root / "fp_analysis" / "per_event.tsv",
        valid_root / "fp_analysis" / "summary.tsv",
        valid_root / "summary.json",
        topk_root / "per_event.tsv",
        topk_root / "summary.tsv",
        topk_root / "paired_transition_summary.tsv",
        topk_root / "run_manifest.json",
        topk_evidence_root / "run_manifest.json",
        extended_root / "summary.tsv",
        extended_root / "run_manifest.json",
        nine_arm_root / "summary.json",
        nine_arm_root / "per_event.tsv",
        oob_v2_root / "oob_contract.json",
        oob_v2_root / "source_valid" / "per_event.tsv",
        oob_v2_root / "invalid_two_arm" / "per_event.tsv",
        oob_v2_root / "invalid_nine_arm" / "per_event.tsv",
        oob_v2_root / "ambiguous_two_arm_r1" / "per_event.tsv",
        oob_v2_root / "ambiguous_two_arm_r2" / "per_event.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    data_root = analysis_root / "data"
    job3 = data_root / "job3_valid_fp"
    job4 = data_root / "job4_support_topk"
    job5 = data_root / "job5_oob_v2"
    shared = data_root / "shared"
    visuals = analysis_root / "visualizations"
    for path in (job3, job4, job5, shared, visuals):
        path.mkdir(parents=True, exist_ok=True)

    _copy_files(
        [
            valid_root / "run_plan.json",
            valid_root / "summary.json",
            valid_root / "fp_analysis" / "summary.json",
            valid_root / "fp_analysis" / "summary.tsv",
            valid_root / "fp_analysis" / "per_event.tsv",
        ],
        job3,
    )
    _copy_files(
        [
            topk_root / "experiment_plan.json",
            topk_root / "run_manifest.json",
            topk_root / "summary.json",
            topk_root / "summary.tsv",
            topk_root / "paired_transition_summary.tsv",
            topk_root / "per_event.tsv",
            topk_evidence_root / "experiment_plan.json",
            topk_evidence_root / "run_manifest.json",
        ],
        job4,
        prefixes={
            topk_evidence_root / "experiment_plan.json": "evidence_",
            topk_evidence_root / "run_manifest.json": "evidence_",
        },
    )
    _copy_oob_v2(oob_v2_root, job5)

    old_valid_rows = _read_tsv(valid_root / "fp_analysis" / "per_event.tsv")
    oob_valid_rows = _read_tsv(oob_v2_root / "source_valid" / "per_event.tsv")
    valid_rows = _merge_valid_oob(old_valid_rows, oob_valid_rows)
    valid_summary = _valid_summary_rows(valid_rows)
    topk_rows = _read_tsv(topk_root / "per_event.tsv")
    topk_summary = _read_tsv(topk_root / "summary.tsv")
    job1_rows = _merge_job1_oob(
        _read_tsv(extended_root / "master_results.tsv"),
        _read_tsv(oob_v2_root / "invalid_two_arm" / "per_event.tsv"),
        _read_tsv(oob_v2_root / "ambiguous_two_arm_r1" / "per_event.tsv"),
        _read_tsv(oob_v2_root / "ambiguous_two_arm_r2" / "per_event.tsv"),
        oob_v2_root / "oob_contract.json",
    )
    job2_rows = _merge_job2_oob(
        _read_tsv(nine_arm_root / "per_event.tsv"),
        _read_tsv(oob_v2_root / "invalid_nine_arm" / "per_event.tsv"),
        oob_v2_root / "oob_contract.json",
    )
    job1_summary, job1_shared, job1_paired, job1_invalid_paired = (
        _summarize_job1(job1_rows, job2_rows)
    )
    job2_summary = _summarize_job2(job2_rows)
    job2_paired = _job2_pair_summaries(job2_rows)
    job2_ablation = _job2_ablation_rows(job2_rows)
    per_metric_matrix = _per_metric_arm_matrix(job2_rows)
    per_metric_readout = _per_metric_readout(
        job1_shared,
        per_metric_matrix,
    )
    oob_rows, collision_rows = _valid_fp_diagnostics(valid_rows, valid_root)
    transitions = _support_topk_transitions(topk_rows, topk_evidence_root)
    inventory = _gt_inventory()
    comparable = _comparability_table()

    stale_oob_diagnostics = shared / "valid_oob_fp_diagnostics.tsv"
    if stale_oob_diagnostics.exists():
        stale_oob_diagnostics.unlink()
    _write_tsv(
        shared / "valid_oob_v2_regression.tsv",
        _oob_v2_regression_rows(oob_v2_root),
    )
    _write_tsv(shared / "valid_collision_fp_diagnostics.tsv", collision_rows)
    _write_tsv(shared / "support_topk_event_transitions.tsv", transitions)
    _write_tsv(shared / "scored_gt_inventory.tsv", inventory)
    _write_tsv(shared / "cross_experiment_comparability.tsv", comparable)
    _write_tsv(shared / "job1_global_vs_local.tsv", job1_shared)
    _write_tsv(shared / "job2_ablation_comparisons.tsv", job2_ablation)
    _write_tsv(shared / "per_metric_arm_matrix.tsv", per_metric_matrix)
    _write_tsv(
        shared / "per_metric_visual_config_readout.tsv",
        per_metric_readout,
    )
    _write_tsv(job3 / "per_event.tsv", valid_rows)
    _write_tsv(job3 / "summary.tsv", valid_summary)
    _write_json(
        job3 / "summary.json",
        {
            "schema_version": "source_valid_fp_with_oob_p0b_v2",
            "oob_contract": _read_json(oob_v2_root / "oob_contract.json"),
            "summary": valid_summary,
        },
    )
    _write_tsv(data_root / "job1" / "per_event.tsv", job1_rows)
    _write_tsv(data_root / "job1" / "master_results.tsv", job1_rows)
    _write_tsv(data_root / "job1" / "summary.tsv", job1_summary)
    _write_tsv(
        data_root / "job1" / "paired_transition_summary.tsv",
        job1_paired,
    )
    _write_tsv(
        data_root / "job1" / "invalid_paired_transition_summary.tsv",
        job1_invalid_paired,
    )
    _write_json(
        data_root / "job1" / "summary.json",
        {
            "schema_version": "cal_dataset1_job1_oob_p0b_v2",
            "oob_contract": _read_json(oob_v2_root / "oob_contract.json"),
            "summary": job1_summary,
        },
    )
    _write_tsv(data_root / "job2" / "per_event.tsv", job2_rows)
    _write_tsv(data_root / "job2" / "policy_summary.tsv", job2_summary)
    _write_tsv(
        data_root / "job2" / "paired_transition_summary.tsv",
        job2_paired,
    )
    _write_json(
        data_root / "job2" / "summary.json",
        {
            "schema_version": "cal_dataset1_visual_config_oob_p0b_v2",
            "oob_contract": _read_json(oob_v2_root / "oob_contract.json"),
            "summary": job2_summary,
        },
    )

    (visuals / "valid-fp-calibration.html").write_text(
        _valid_fp_html(
            valid_summary,
            _oob_v2_regression_rows(oob_v2_root),
            collision_rows,
        ),
        encoding="utf-8",
    )
    (visuals / "support-topk-invalid.html").write_text(
        _support_topk_html(
            topk_rows,
            topk_evidence_root,
            visuals / "support-topk-invalid.html",
            gt_label="invalid",
            title="Support TopK · scored invalid events",
        ),
        encoding="utf-8",
    )
    (visuals / "support-topk-ambiguous-review.html").write_text(
        _support_topk_html(
            topk_rows,
            topk_evidence_root,
            visuals / "support-topk-ambiguous-review.html",
            gt_label="ambiguous",
            title="Support TopK · ambiguous manual review",
        ),
        encoding="utf-8",
    )
    (visuals / "completed-calibration-dashboard.html").write_text(
        _dashboard_html(valid_summary, topk_summary, job1_summary),
        encoding="utf-8",
    )
    _write_hash_inventory(data_root, shared / "copied_files.sha256")
    print(json.dumps({
        "analysis_root": str(analysis_root),
        "valid_events": len(valid_rows),
        "valid_false_positives": sum(
            row.get("false_positive") == "1" for row in valid_rows
        ),
        "support_topk_results": len(topk_rows),
        "oob_fp_diagnostics": len(oob_rows),
        "oob_contract": "oob_p0b_v2",
        "collision_fp_diagnostics": len(collision_rows),
        "visualizations": [
            str(path)
            for path in sorted(visuals.glob("*.html"))
        ],
    }, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    parser.add_argument(
        "--valid-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_valid_fp_gpt56"
        ),
    )
    parser.add_argument(
        "--topk-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_support_topk_gpt56"
        ),
    )
    parser.add_argument(
        "--topk-evidence-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_support_topk_evidence"
        ),
    )
    parser.add_argument(
        "--extended-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_1_extended_gpt56"
        ),
    )
    parser.add_argument(
        "--nine-arm-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_1_visual_config_gpt56"
        ),
    )
    parser.add_argument(
        "--oob-v2-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp3_oob_v2_gpt56"
        ),
    )
    return parser.parse_args()


def _copy_files(
    sources: Iterable[Path],
    destination: Path,
    *,
    prefixes: dict[Path, str] | None = None,
) -> None:
    prefixes = prefixes or {}
    for source in sources:
        if not source.is_file():
            continue
        prefix = prefixes.get(source, "")
        shutil.copy2(source, destination / f"{prefix}{source.name}")


def _copy_oob_v2(source_root: Path, destination_root: Path) -> None:
    _copy_files(
        [source_root / "oob_contract.json", source_root / "preflight.json"],
        destination_root,
    )
    for run_name in (
        "source_valid",
        "invalid_two_arm",
        "invalid_nine_arm",
        "ambiguous_two_arm_r1",
        "ambiguous_two_arm_r2",
    ):
        destination = destination_root / run_name
        destination.mkdir(parents=True, exist_ok=True)
        _copy_files(
            sorted((source_root / run_name).glob("*")),
            destination,
        )


def _merge_valid_oob(
    old_rows: list[dict[str, str]],
    oob_v2_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        dict(row) for row in old_rows if row.get("metric") != "oob"
    ]
    for row in oob_v2_rows:
        predicted = row.get("predicted_label", "")
        result.append({
            "case_id": row["case_id"],
            "metric": "oob",
            "event_id": row["event_id"],
            "gt_label": "valid",
            "predicted_label": predicted,
            "resolved": row.get("resolved", "0"),
            "false_positive": int(predicted == "invalid"),
            "route": row.get("route", ""),
            "requires_vlm": row.get("requires_vlm", "0"),
            "vlm_adjudicated": row.get("judge_called", "0"),
            "adjudication_error": row.get("error", ""),
            "geometry_warning_excluded": "0",
        })
    return sorted(
        result,
        key=lambda row: (
            str(row["case_id"]),
            str(row["metric"]),
            str(row["event_id"]),
        ),
    )


def _valid_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", rows)
    ]
    groups.extend(
        (
            "metric",
            metric,
            [row for row in rows if row["metric"] == metric],
        )
        for metric in ("collision", "oob", "support")
    )
    groups.extend(
        (
            "route",
            route,
            [row for row in rows if row["route"] == route],
        )
        for route in sorted({str(row["route"]) for row in rows})
    )
    result: list[dict[str, Any]] = []
    for group_type, group, selected in groups:
        resolved = [row for row in selected if str(row["resolved"]) == "1"]
        false_positives = [
            row for row in selected if str(row["false_positive"]) == "1"
        ]
        vlm = [
            row for row in selected if str(row["vlm_adjudicated"]) == "1"
        ]
        vlm_resolved = [row for row in vlm if str(row["resolved"]) == "1"]
        vlm_fp = [
            row for row in vlm if str(row["false_positive"]) == "1"
        ]
        total = len(selected)
        result.append({
            "group_type": group_type,
            "group": group,
            "valid_gt_total": total,
            "resolved": len(resolved),
            "unresolved": total - len(resolved),
            "predicted_valid": total - len(false_positives),
            "false_positives": len(false_positives),
            "fp_rate_all": len(false_positives) / total if total else None,
            "fp_rate_resolved": (
                len(false_positives) / len(resolved) if resolved else None
            ),
            "specificity_all": (
                1.0 - len(false_positives) / total if total else None
            ),
            "specificity_resolved": (
                1.0 - len(false_positives) / len(resolved)
                if resolved else None
            ),
            "vlm_adjudicated": len(vlm),
            "vlm_resolved": len(vlm_resolved),
            "vlm_false_positives": len(vlm_fp),
            "vlm_fp_rate_all": len(vlm_fp) / len(vlm) if vlm else None,
            "vlm_fp_rate_resolved": (
                len(vlm_fp) / len(vlm_resolved) if vlm_resolved else None
            ),
        })
    return result


def _replace_from_template(
    template: dict[str, str],
    replacement: dict[str, str],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(template)
    for field in (
        "predicted_label",
        "resolved",
        "match",
        "confidence",
        "reason",
        "elapsed_seconds",
        "error",
    ):
        if field in replacement:
            row[field] = replacement[field]
    row["contract_sha256"] = contract_sha256
    return row


def _merge_job1_oob(
    old_rows: list[dict[str, str]],
    invalid_rows: list[dict[str, str]],
    ambiguous_r1: list[dict[str, str]],
    ambiguous_r2: list[dict[str, str]],
    contract_path: Path,
) -> list[dict[str, Any]]:
    template = {
        (
            row["stratum"],
            int(row["repeat"]),
            row["case_id"],
            row["event_id"],
            row["arm"],
        ): row
        for row in old_rows
        if row["metric"] == "oob"
    }
    result: list[dict[str, Any]] = [
        dict(row) for row in old_rows if row["metric"] != "oob"
    ]
    contract_sha = _file_sha256(contract_path)
    runs = [
        ("invalid", 1, invalid_rows),
        ("ambiguous", 1, ambiguous_r1),
        ("ambiguous", 2, ambiguous_r2),
    ]
    for stratum, repeat, rows in runs:
        for replacement in rows:
            key = (
                stratum,
                repeat,
                replacement["case_id"],
                replacement["event_id"],
                replacement["arm"],
            )
            if key not in template:
                raise KeyError(f"missing Job 1 OOB template: {key}")
            result.append(
                _replace_from_template(
                    template[key],
                    replacement,
                    contract_sha256=contract_sha,
                )
            )
    return sorted(
        result,
        key=lambda row: (
            0 if row["stratum"] == "invalid" else 1,
            int(row["repeat"]),
            str(row["case_id"]),
            str(row["metric"]),
            str(row["event_id"]),
            str(row["arm"]),
        ),
    )


def _merge_job2_oob(
    old_rows: list[dict[str, str]],
    oob_v2_rows: list[dict[str, str]],
    contract_path: Path,
) -> list[dict[str, Any]]:
    template = {
        (row["case_id"], row["event_id"], row["arm"]): row
        for row in old_rows
        if row["metric"] == "oob"
    }
    result: list[dict[str, Any]] = [
        dict(row) for row in old_rows if row["metric"] != "oob"
    ]
    contract_sha = _file_sha256(contract_path)
    for replacement in oob_v2_rows:
        key = (
            replacement["case_id"],
            replacement["event_id"],
            replacement["arm"],
        )
        if key not in template:
            raise KeyError(f"missing Job 2 OOB template: {key}")
        result.append(
            _replace_from_template(
                template[key],
                replacement,
                contract_sha256=contract_sha,
            )
        )
    return sorted(
        result,
        key=lambda row: (
            str(row["case_id"]),
            str(row["metric"]),
            str(row["event_id"]),
            str(row["arm"]),
        ),
    )


def _job1_prior_invalid_repeat(
    job1_rows: list[dict[str, Any]],
    job2_rows: list[dict[str, Any]],
    *,
    metric: str,
    arm: str,
) -> tuple[int, int]:
    job2_arm = {
        "fixed_global_highlight": "fixed_global_highlight",
        "metric_local_highlight": "presence_local_raw_highlight",
    }[arm]
    left = {
        (row["case_id"], row["event_id"]): row
        for row in job1_rows
        if row["stratum"] == "invalid"
        and row["metric"] == metric
        and row["arm"] == arm
    }
    right = {
        (row["case_id"], row["event_id"]): row
        for row in job2_rows
        if row["metric"] == metric and row["arm"] == job2_arm
    }
    if set(left) != set(right):
        raise RuntimeError(f"Job 1/2 repeat keys differ for {metric} {arm}")
    return len(left), sum(
        left[key]["predicted_label"] == right[key]["predicted_label"]
        for key in left
    )


def _summarize_job1(
    rows: list[dict[str, Any]],
    job2_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    arms = ("fixed_global_highlight", "metric_local_highlight")
    metrics = ("collision", "oob", "support")
    summary: list[dict[str, Any]] = []
    by_arm_metric: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in arms:
        for group_type, group in [
            ("overall", "overall"),
            *(("metric", metric) for metric in metrics),
        ]:
            invalid = [
                row for row in rows
                if row["stratum"] == "invalid"
                and row["arm"] == arm
                and (group_type == "overall" or row["metric"] == group)
            ]
            ambiguous_runs = {
                repeat: [
                    row for row in rows
                    if row["stratum"] == "ambiguous"
                    and int(row["repeat"]) == repeat
                    and row["arm"] == arm
                    and (group_type == "overall" or row["metric"] == group)
                ]
                for repeat in (1, 2)
            }
            ambiguous = ambiguous_runs[1] + ambiguous_runs[2]
            left = {
                (row["case_id"], row["metric"], row["event_id"]): row
                for row in ambiguous_runs[1]
            }
            right = {
                (row["case_id"], row["metric"], row["event_id"]): row
                for row in ambiguous_runs[2]
            }
            if set(left) != set(right):
                raise RuntimeError(f"ambiguous keys differ for {arm} {group}")
            ambiguous_agreements = sum(
                left[key]["predicted_label"] == right[key]["predicted_label"]
                for key in left
            )
            prior_pairs = prior_agreements = 0
            selected_metrics = metrics if group_type == "overall" else (group,)
            for metric in selected_metrics:
                pairs, agreements = _job1_prior_invalid_repeat(
                    rows,
                    job2_rows,
                    metric=metric,
                    arm=arm,
                )
                prior_pairs += pairs
                prior_agreements += agreements
            invalid_detected = sum(
                row["predicted_label"] == "invalid" for row in invalid
            )
            ambiguous_invalid = sum(
                row["predicted_label"] == "invalid" for row in ambiguous
            )
            record = {
                "arm": arm,
                "group_type": group_type,
                "group": group,
                "invalid_events": len(invalid),
                "invalid_detected": invalid_detected,
                "invalid_recall": invalid_detected / len(invalid),
                "ambiguous_events": len(left),
                "ambiguous_repeat_count": 2,
                "ambiguous_judgements": len(ambiguous),
                "ambiguous_predicted_invalid": ambiguous_invalid,
                "ambiguous_invalid_rate": ambiguous_invalid / len(ambiguous),
                "ambiguous_exact_input_repeat_pairs": len(left),
                "ambiguous_repeat_agreement": (
                    ambiguous_agreements / len(left)
                ),
                "prior_invalid_exact_input_repeat_pairs": prior_pairs,
                "prior_invalid_repeat_agreements": prior_agreements,
                "combined_exact_input_repeat_pairs": prior_pairs + len(left),
                "combined_repeat_agreements": (
                    prior_agreements + ambiguous_agreements
                ),
                "combined_repeat_agreement": (
                    (prior_agreements + ambiguous_agreements)
                    / (prior_pairs + len(left))
                ),
                "total_unique_events": len(invalid) + len(left),
            }
            summary.append(record)
            by_arm_metric[(arm, group)] = record

    shared: list[dict[str, Any]] = []
    for group_type, group in [
        ("overall", "overall"),
        *(("metric", metric) for metric in metrics),
    ]:
        fixed = by_arm_metric[("fixed_global_highlight", group)]
        local = by_arm_metric[("metric_local_highlight", group)]
        shared.append({
            "group_type": group_type,
            "group": group,
            "total_unique_events": fixed["total_unique_events"],
            "invalid_events": fixed["invalid_events"],
            "fixed_invalid_detected": fixed["invalid_detected"],
            "fixed_invalid_recall": f"{fixed['invalid_recall']:.4f}",
            "local_invalid_detected": local["invalid_detected"],
            "local_invalid_recall": f"{local['invalid_recall']:.4f}",
            "local_minus_fixed_invalid_recall_pp": (
                f"{(local['invalid_recall']-fixed['invalid_recall'])*100:.2f}"
            ),
            "ambiguous_events": fixed["ambiguous_events"],
            "ambiguous_repeats": 2,
            "ambiguous_judgements_per_arm": fixed["ambiguous_judgements"],
            "fixed_ambiguous_predicted_invalid": (
                fixed["ambiguous_predicted_invalid"]
            ),
            "fixed_ambiguous_invalid_rate": (
                f"{fixed['ambiguous_invalid_rate']:.4f}"
            ),
            "local_ambiguous_predicted_invalid": (
                local["ambiguous_predicted_invalid"]
            ),
            "local_ambiguous_invalid_rate": (
                f"{local['ambiguous_invalid_rate']:.4f}"
            ),
            "local_minus_fixed_ambiguous_invalid_rate_pp": (
                f"{(local['ambiguous_invalid_rate']-fixed['ambiguous_invalid_rate'])*100:.2f}"
            ),
            "fixed_ambiguous_repeat_agreement": (
                f"{fixed['ambiguous_repeat_agreement']:.4f}"
            ),
            "local_ambiguous_repeat_agreement": (
                f"{local['ambiguous_repeat_agreement']:.4f}"
            ),
            "fixed_combined_repeat_agreements": (
                fixed["combined_repeat_agreements"]
            ),
            "fixed_combined_repeat_pairs": (
                fixed["combined_exact_input_repeat_pairs"]
            ),
            "fixed_combined_repeat_agreement": (
                f"{fixed['combined_repeat_agreement']:.4f}"
            ),
            "local_combined_repeat_agreements": (
                local["combined_repeat_agreements"]
            ),
            "local_combined_repeat_pairs": (
                local["combined_exact_input_repeat_pairs"]
            ),
            "local_combined_repeat_agreement": (
                f"{local['combined_repeat_agreement']:.4f}"
            ),
        })

    paired: list[dict[str, Any]] = []
    invalid_paired: list[dict[str, Any]] = []
    for group_type, group, predicate in [
        ("overall", "overall", lambda row: True),
        *((
            "metric",
            metric,
            lambda row, metric=metric: row["metric"] == metric,
        ) for metric in metrics),
        ("severity", "obvious", lambda row: row["severity_class"] == "obvious"),
        ("severity", "subtle", lambda row: row["severity_class"] == "subtle"),
    ]:
        invalid_transitions = _paired_arm_counts(
            [
                row for row in rows
                if row["stratum"] == "invalid" and predicate(row)
            ],
            "fixed_global_highlight",
            "metric_local_highlight",
        )
        if group_type != "severity":
            ambiguous_transitions = _paired_arm_counts(
                [
                    row for row in rows
                    if row["stratum"] == "ambiguous" and predicate(row)
                ],
                "fixed_global_highlight",
                "metric_local_highlight",
                repeat_sensitive=True,
            )
            paired.append({
                "group_type": group_type,
                "group": group,
                "invalid_pairs": invalid_transitions["total"],
                "invalid_both_correct": invalid_transitions["both_invalid"],
                "invalid_fixed_only_correct": invalid_transitions["left_only_invalid"],
                "invalid_local_only_correct": invalid_transitions["right_only_invalid"],
                "invalid_both_incorrect": invalid_transitions["both_valid"],
                "ambiguous_pairwise_judgements": ambiguous_transitions["total"],
                "ambiguous_both_invalid": ambiguous_transitions["both_invalid"],
                "ambiguous_both_valid": ambiguous_transitions["both_valid"],
                "ambiguous_fixed_invalid_local_valid": ambiguous_transitions["left_only_invalid"],
                "ambiguous_fixed_valid_local_invalid": ambiguous_transitions["right_only_invalid"],
                "ambiguous_arm_verdict_agreement": (
                    (ambiguous_transitions["both_invalid"] + ambiguous_transitions["both_valid"])
                    / ambiguous_transitions["total"]
                ),
            })
        invalid_paired.append({
            "group_type": group_type,
            "group": group,
            "total_pairs": invalid_transitions["total"],
            "both_correct": invalid_transitions["both_invalid"],
            "fixed_only_correct": invalid_transitions["left_only_invalid"],
            "local_only_correct": invalid_transitions["right_only_invalid"],
            "both_incorrect": invalid_transitions["both_valid"],
            "unresolved_pairs": 0,
            "local_minus_fixed_correct": (
                invalid_transitions["right_only_invalid"]
                - invalid_transitions["left_only_invalid"]
            ),
        })
    return summary, shared, paired, invalid_paired


def _paired_arm_counts(
    rows: list[dict[str, Any]],
    left_arm: str,
    right_arm: str,
    *,
    repeat_sensitive: bool = False,
) -> dict[str, int]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        base = (row["case_id"], row["metric"], row["event_id"])
        return base + ((int(row["repeat"]),) if repeat_sensitive else ())

    left = {key(row): row for row in rows if row["arm"] == left_arm}
    right = {key(row): row for row in rows if row["arm"] == right_arm}
    if set(left) != set(right):
        raise RuntimeError(f"paired arm keys differ: {left_arm} vs {right_arm}")
    counts = {
        "total": len(left),
        "both_invalid": 0,
        "left_only_invalid": 0,
        "right_only_invalid": 0,
        "both_valid": 0,
    }
    for event_key in left:
        left_invalid = left[event_key]["predicted_label"] == "invalid"
        right_invalid = right[event_key]["predicted_label"] == "invalid"
        if left_invalid and right_invalid:
            counts["both_invalid"] += 1
        elif left_invalid:
            counts["left_only_invalid"] += 1
        elif right_invalid:
            counts["right_only_invalid"] += 1
        else:
            counts["both_valid"] += 1
    return counts


def _summarize_job2(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    arms = sorted({str(row["arm"]) for row in rows})
    result: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        groups = [("overall", "overall", arm_rows)]
        groups.extend(
            (
                "metric",
                metric,
                [row for row in arm_rows if row["metric"] == metric],
            )
            for metric in ("collision", "oob", "support")
        )
        groups.extend(
            (
                "severity",
                severity,
                [
                    row for row in arm_rows
                    if row["severity_class"] == severity
                ],
            )
            for severity in ("obvious", "subtle")
        )
        for group_type, group, selected in groups:
            resolved = [row for row in selected if str(row["resolved"]) == "1"]
            correct = [
                row for row in resolved if row["predicted_label"] == "invalid"
            ]
            confidences = [
                float(row["confidence"]) for row in resolved
                if str(row.get("confidence", "")).strip()
            ]
            latencies = [
                float(row["elapsed_seconds"]) for row in resolved
                if str(row.get("elapsed_seconds", "")).strip()
            ]
            budgets = [
                float(row["image_budget"]) for row in resolved
                if str(row.get("image_budget", "")).strip()
            ]
            result.append({
                "arm": arm,
                "group_type": group_type,
                "group": group,
                "total": len(selected),
                "resolved": len(resolved),
                "coverage": len(resolved) / len(selected),
                "correct": len(correct),
                "invalid_detected": len(correct),
                "false_negatives": len(selected) - len(correct),
                "invalid_recall_all": len(correct) / len(selected),
                "invalid_recall_resolved": (
                    len(correct) / len(resolved) if resolved else None
                ),
                "mean_confidence": (
                    sum(confidences) / len(confidences)
                    if confidences else None
                ),
                "mean_latency_seconds": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
                "mean_image_budget": (
                    sum(budgets) / len(budgets) if budgets else None
                ),
                "error_count": len(selected) - len(resolved),
            })
    return result


JOB2_COMPARISONS = (
    ("global_highlight_effect", "fixed_global", "fixed_global_highlight"),
    ("local_highlight_effect", "presence_local_raw", "presence_local_raw_highlight"),
    ("global_context_with_raw", "presence_local_raw", "presence_global_local_raw"),
    ("global_context_with_highlight", "presence_local_raw_highlight", "deterministic_metric_local"),
    ("global_first_to_local_first", "deterministic_metric_local", "order_local_first_full"),
    ("full_to_compact_global_first", "deterministic_metric_local", "budget_global_first_compact"),
    ("full_to_compact_local_first", "order_local_first_full", "budget_local_first_compact"),
    ("fixed_global_to_local_raw", "fixed_global", "presence_local_raw"),
    ("fixed_global_to_full_local", "fixed_global", "deterministic_metric_local"),
)


def _job2_pair_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for comparison, left_arm, right_arm in JOB2_COMPARISONS:
        for group_type, group, predicate in [
            ("overall", "overall", lambda row: True),
            *((
                "metric",
                metric,
                lambda row, metric=metric: row["metric"] == metric,
            ) for metric in ("collision", "oob", "support")),
            ("severity", "obvious", lambda row: row["severity_class"] == "obvious"),
            ("severity", "subtle", lambda row: row["severity_class"] == "subtle"),
        ]:
            counts = _paired_arm_counts(
                [row for row in rows if predicate(row)],
                left_arm,
                right_arm,
            )
            result.append({
                "comparison": comparison,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "group_type": group_type,
                "group": group,
                "total_pairs": counts["total"],
                "both_correct": counts["both_invalid"],
                "left_only_correct": counts["left_only_invalid"],
                "right_only_correct": counts["right_only_invalid"],
                "both_incorrect": counts["both_valid"],
                "unresolved": 0,
                "right_minus_left_correct": (
                    counts["right_only_invalid"] - counts["left_only_invalid"]
                ),
            })
    return result


def _job2_ablation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    questions = [
        ("local_vs_global", "fixed_global", "presence_local_raw",
         "local raw is the strongest two-image baseline"),
        ("highlight_global", "fixed_global", "fixed_global_highlight",
         "duplicated highlight does not improve invalid recall"),
        ("highlight_local", "presence_local_raw", "presence_local_raw_highlight",
         "highlight is neutral-to-negative when it duplicates the same poses"),
        ("global_context_raw", "presence_local_raw", "presence_global_local_raw",
         "global context is not universally required"),
        ("global_context_highlight", "presence_local_raw_highlight", "deterministic_metric_local",
         "full highlighted context ties the local-highlight arm"),
        ("image_order_full", "deterministic_metric_local", "order_local_first_full",
         "local-first is better overall after OOB v2 replacement"),
        ("budget_global_first", "deterministic_metric_local", "budget_global_first_compact",
         "global-first compact loses one correct decision"),
        ("budget_local_first", "order_local_first_full", "budget_local_first_compact",
         "compact local-first loses one overall despite stronger OOB"),
    ]
    by_arm = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in {str(row["arm"]) for row in rows}
    }
    result = []
    for question, left_arm, right_arm, interpretation in questions:
        left = by_arm[left_arm]
        right = by_arm[right_arm]
        counts = _paired_arm_counts(rows, left_arm, right_arm)
        left_correct = sum(row["predicted_label"] == "invalid" for row in left)
        right_correct = sum(row["predicted_label"] == "invalid" for row in right)
        result.append({
            "question": question,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_images": left[0]["image_budget"],
            "right_images": right[0]["image_budget"],
            "left_correct": left_correct,
            "right_correct": right_correct,
            "left_recall": f"{left_correct/len(left):.4f}",
            "right_recall": f"{right_correct/len(right):.4f}",
            "right_minus_left_pp": (
                f"{(right_correct/len(right)-left_correct/len(left))*100:.2f}"
            ),
            "paired_left_only": counts["left_only_invalid"],
            "paired_right_only": counts["right_only_invalid"],
            "interpretation": interpretation,
        })
    return result


def _per_metric_arm_matrix(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = {
        "fixed_global": ("Global raw", 2, 2, "raw_once_per_pose"),
        "fixed_global_highlight": ("Global raw + highlight", 2, 4, "raw_plus_highlight_duplicate"),
        "presence_local_raw": ("Local raw", 2, 2, "raw_once_per_pose"),
        "presence_local_raw_highlight": ("Local raw + highlight", 2, 4, "raw_plus_highlight_duplicate"),
        "presence_global_local_raw": ("Global top + local raw", 3, 3, "mixed_no_local_duplicate"),
        "deterministic_metric_local": ("Full global-first", 3, 5, "raw_plus_highlight_duplicate"),
        "order_local_first_full": ("Full local-first", 3, 5, "raw_plus_highlight_duplicate"),
        "budget_global_first_compact": ("Compact global-first", 2, 3, "raw_plus_highlight_duplicate"),
        "budget_local_first_compact": ("Compact local-first", 2, 3, "raw_plus_highlight_duplicate"),
    }
    result = []
    for arm in metadata:
        selected = [row for row in rows if row["arm"] == arm]
        recalls = {}
        for metric in ("collision", "oob", "support"):
            metric_rows = [row for row in selected if row["metric"] == metric]
            recalls[metric] = (
                sum(row["predicted_label"] == "invalid" for row in metric_rows)
                / len(metric_rows)
            )
        label, poses, images, presentation = metadata[arm]
        result.append({
            "arm": arm,
            "short_label": label,
            "unique_poses": poses,
            "image_count": images,
            "presentation": presentation,
            "collision_recall": f"{recalls['collision']:.3f}",
            "oob_recall": f"{recalls['oob']:.3f}",
            "support_recall": f"{recalls['support']:.3f}",
        })
    return result


def _per_metric_readout(
    job1_shared: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    job1 = {
        row["group"]: row
        for row in job1_shared
        if row["group_type"] == "metric"
    }
    notes = {
        "collision": (
            "Local evidence remains the efficient baseline. Remaining errors "
            "are dominated by the judgement/GT boundary."
        ),
        "oob": (
            "Under oob_p0b_v2, local raw/local highlight/compact local-first "
            "reach 8/8; global context is optional and does not improve recall."
        ),
        "support": (
            "Support benefits most from distinct local angles; preserve the "
            "larger local packet rather than raw/highlight duplication."
        ),
    }
    result = []
    for metric, ambiguous_n in (
        ("collision", 4),
        ("oob", 5),
        ("support", 6),
    ):
        score_field = f"{metric}_recall"
        best = max(matrix, key=lambda row: float(row[score_field]))
        source = job1[metric]
        result.append({
            "metric": metric,
            "invalid_n": 8,
            "ambiguous_n": ambiguous_n,
            "total_unique_events": 8 + ambiguous_n,
            "best_job2_arm": best["arm"],
            "best_job2_invalid_recall": (
                f"{float(best[score_field]):.4f}"
            ),
            "best_observed_budget": best["image_count"],
            "fixed_job1_invalid_recall": source["fixed_invalid_recall"],
            "local_job1_invalid_recall": source["local_invalid_recall"],
            "fixed_ambiguous_invalid_rate": source["fixed_ambiguous_invalid_rate"],
            "local_ambiguous_invalid_rate": source["local_ambiguous_invalid_rate"],
            "fixed_combined_repeat_agreement": source["fixed_combined_repeat_agreement"],
            "local_combined_repeat_agreement": source["local_combined_repeat_agreement"],
            "pilot_readout": notes[metric],
            "exact_current_policy_tested": "No",
        })
    return result


def _oob_v2_regression_rows(root: Path) -> list[dict[str, Any]]:
    contract = _read_json(root / "oob_contract.json")
    valid = _read_tsv(root / "source_valid" / "summary.tsv")[0]
    invalid_two = _read_tsv(root / "invalid_two_arm" / "summary.tsv")
    invalid_nine = _read_tsv(root / "invalid_nine_arm" / "summary.tsv")
    return [{
        "evaluator_version": contract.get("evaluator_version"),
        "valid_gt_total": valid["valid_gt"],
        "false_positives_before": 24,
        "false_positives_after": valid["false_positives"],
        "specificity_after": valid["specificity"],
        "direct_valid_after": valid["direct_valid"],
        "vlm_calls_after": valid["vlm_calls"],
        "floor_contact_tolerance_m": contract.get("floor_contact_tolerance_m"),
        "two_arm_fixed_invalid_recall": next(
            row["invalid_recall"]
            for row in invalid_two
            if row["group_type"] == "arm"
            and row["group"] == "fixed_global_highlight"
        ),
        "two_arm_local_invalid_recall": next(
            row["invalid_recall"]
            for row in invalid_two
            if row["group_type"] == "arm"
            and row["group"] == "metric_local_highlight"
        ),
        "nine_arm_best_invalid_recall": max(
            float(row["invalid_recall"])
            for row in invalid_nine
            if row["group_type"] == "arm"
        ),
    }]


def _valid_fp_diagnostics(
    valid_rows: list[dict[str, str]],
    valid_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: dict[str, dict[str, Any]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    oob: list[dict[str, Any]] = []
    collision: list[dict[str, Any]] = []
    for row in valid_rows:
        if row.get("false_positive") != "1":
            continue
        case_id = row["case_id"]
        if case_id not in reports:
            reports[case_id] = _read_json(
                valid_root
                / "deterministic"
                / case_id
                / "evaluation_report.json"
            )
            scenes[case_id] = _read_json(
                PROJECT_ROOT
                / "Support"
                / "datasets"
                / "cal_dataset1"
                / "fixtures"
                / case_id
                / "generated_scene.json"
            )
        scene_objects = {
            str(item.get("id")): item
            for item in scenes[case_id].get("objects") or []
            if isinstance(item, dict)
        }
        if row["metric"] == "oob":
            record = next(
                item
                for item in reports[case_id]["metrics"]["oob"]["objects"]
                if str(item.get("object_id")) == row["event_id"]
            )
            intervals = record.get("obb_intervals") or {}
            flags = record.get("plane_flags") or {}
            floor_depth = max(0.0, -float((intervals.get("z") or [0.0])[0]))
            object_record = scene_objects.get(row["event_id"], {})
            oob.append({
                "case_id": case_id,
                "event_id": row["event_id"],
                "category": object_record.get("category"),
                "active_plane_flags": ",".join(
                    key for key, value in flags.items() if value
                ),
                "floor_penetration_m": floor_depth,
                "floor_penetration_mm": round(floor_depth * 1000.0, 6),
                "final_verdict": record.get("final_verdict"),
                "confidence": (record.get("judge_result") or {}).get("confidence"),
                "reason": (record.get("judge_result") or {}).get("reason"),
            })
        elif row["metric"] == "collision":
            left, right = row["event_id"].split("|", 1)
            record = next(
                item
                for item in reports[case_id]["metrics"]["collision"]["pairs"]
                if {
                    str(item.get("object_a")),
                    str(item.get("object_b")),
                } == {left, right}
            )
            collision.append({
                "case_id": case_id,
                "event_id": row["event_id"],
                "categories": " | ".join(
                    str(scene_objects.get(object_id, {}).get("category") or object_id)
                    for object_id in (left, right)
                ),
                "evidence_level": record.get("evidence_level"),
                "mesh_state": (record.get("mesh_evidence") or {}).get("mesh_state"),
                "obb_overlap_depth_m": (
                    record.get("obb_evidence") or {}
                ).get("minimum_overlap_depth_proxy_m"),
                "mesh_intersection_definitive": (
                    (record.get("mesh_evidence") or {}).get("intersection") or {}
                ).get("definitive"),
                "confidence": (record.get("judge_result") or {}).get("confidence"),
                "reason": (record.get("judge_result") or {}).get("reason"),
            })
    return oob, collision


def _support_topk_transitions(
    rows: list[dict[str, str]],
    evidence_root: Path,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["case_id"], row["event_id"])][row["arm"]] = row
    result: list[dict[str, Any]] = []
    for (case_id, event_id), arms in sorted(grouped.items()):
        manifest_path = (
            evidence_root
            / "cases"
            / case_id
            / "events"
            / f"support__{_safe_name(event_id)}"
            / "comparison_manifest.json"
        )
        manifest = _read_json(manifest_path)
        result.append({
            "case_id": case_id,
            "event_id": event_id,
            "gt_label": manifest.get("semantic_label"),
            "severity": manifest.get("severity_class"),
            "top1": arms["support_top1"]["predicted_label"],
            "top2": arms["support_top2"]["predicted_label"],
            "top3": arms["support_top3"]["predicted_label"],
            "top1_confidence": arms["support_top1"]["confidence"],
            "top2_confidence": arms["support_top2"]["confidence"],
            "top3_confidence": arms["support_top3"]["confidence"],
            "selected_top3_view_ids": ",".join(
                str(value)
                for value in (manifest.get("selection") or {}).get(
                    "selected_top3_view_ids", []
                )
            ),
            "top2_to_top3_flip": int(
                arms["support_top2"]["predicted_label"]
                != arms["support_top3"]["predicted_label"]
            ),
        })
    return result


def _gt_inventory() -> list[dict[str, Any]]:
    root = PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"
    cases = _read_json(root / "cases.json").get("cases") or []
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for case in cases:
        gt = _read_json(root / str(case["fixture_dir"]) / "event_gt.json")
        for event in gt.get("events") or []:
            counts[(
                str(case.get("split")),
                str(case.get("review_status")),
                str(event.get("semantic_label")),
                str(event.get("gt_basis")),
            )] += 1
    return [
        {
            "split": key[0],
            "review_status": key[1],
            "semantic_label": key[2],
            "gt_basis": key[3],
            "event_count": count,
            "included_in_accuracy": int(key[2] in {"valid", "invalid"}),
        }
        for key, count in sorted(counts.items())
    ]


def _comparability_table() -> list[dict[str, Any]]:
    return [
        {
            "comparison": "source-valid FP vs Exp1.1 fixed-global invalid",
            "same_gt_family": 0,
            "same_visual_config": 0,
            "formal_combined_accuracy_allowed": 0,
            "use": "diagnostic specificity and recall reported separately",
        },
        {
            "comparison": "source-valid FP vs Exp1.1 metric-local invalid",
            "same_gt_family": 0,
            "same_visual_config": 0,
            "formal_combined_accuracy_allowed": 0,
            "use": "diagnostic specificity and recall reported separately",
        },
        {
            "comparison": "Support Top1 vs Top2 vs Top3",
            "same_gt_family": 1,
            "same_visual_config": 1,
            "formal_combined_accuracy_allowed": 1,
            "use": "paired invalid recall; ambiguous tendency excluded",
        },
        {
            "comparison": "Ambiguous Support TopK",
            "same_gt_family": 1,
            "same_visual_config": 1,
            "formal_combined_accuracy_allowed": 0,
            "use": "manual review and verdict tendency only",
        },
    ]


def _valid_fp_html(
    summary: list[dict[str, str]],
    oob_regression: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
) -> str:
    metric_rows = [
        row for row in summary if row["group_type"] == "metric"
    ]
    bars = "".join(
        _bar(
            row["group"].upper(),
            float(row["fp_rate_all"]) * 100.0,
            f"{row['false_positives']}/{row['valid_gt_total']}",
        )
        for row in metric_rows
    )
    overall = next(
        row for row in summary if row["group_type"] == "overall"
    )
    routed = next(
        row for row in summary
        if row["group_type"] == "route" and row["group"] == "vlm_adjudicated"
    )
    direct_total = (
        int(overall["valid_gt_total"]) - int(overall["vlm_adjudicated"])
    )
    direct_fp = (
        int(overall["false_positives"])
        - int(overall["vlm_false_positives"])
    )
    oob_table = _table(
        [
            "evaluator_version",
            "valid_gt_total",
            "false_positives_before",
            "false_positives_after",
            "specificity_after",
            "direct_valid_after",
            "vlm_calls_after",
            "floor_contact_tolerance_m",
        ],
        oob_regression,
    )
    collision_table = _table(
        [
            "case_id",
            "event_id",
            "categories",
            "obb_overlap_depth_m",
            "confidence",
            "reason",
        ],
        collision_rows,
    )
    body = f"""
<header><h1>Source-valid false-positive calibration</h1>
<p>200/200 resolved · {overall['false_positives']} frozen-label FP · OOB replaced by oob_p0b_v2 · ambiguous excluded</p></header>
<section class="stats">
  <div><b>{float(overall['fp_rate_all'])*100:.1f}%</b><span>overall FP · {overall['false_positives']}/{overall['valid_gt_total']}</span></div>
  <div><b>{float(routed['fp_rate_all'])*100:.1f}%</b><span>VLM-routed FP · {routed['false_positives']}/{routed['valid_gt_total']}</span></div>
  <div><b>{(direct_fp/direct_total)*100:.1f}%</b><span>direct-result FP · {direct_fp}/{direct_total}</span></div>
</section>
<section><h2>FP rate by metric</h2><div class="bars">{bars}</div></section>
<section><h2>OOB v2 regression</h2>
<p>The explicit 5 mm floor-contact tolerance removes all 24 prior shallow-floor
false positives. All 50 source-valid OOB events now resolve directly with no
VLM calls. The invalid recall result is reported separately because it uses
constructed-invalid inputs and different VisualConfig arms.</p>{oob_table}</section>
<section><h2>Collision frozen-label conflicts</h2>
<p>Three routed mesh-uncertain pairs are predicted invalid. They remain counted
as FP under the frozen GT and require separate geometry audit.</p>{collision_table}</section>
"""
    return _standalone("Source-valid FP calibration", body)


def _support_topk_html(
    rows: list[dict[str, str]],
    evidence_root: Path,
    output_path: Path,
    *,
    gt_label: str,
    title: str,
) -> str:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["gt_label"] == gt_label:
            grouped[(row["case_id"], row["event_id"])][row["arm"]] = row
    cards: list[str] = []
    for (case_id, event_id), arm_rows in sorted(grouped.items()):
        manifest_path = (
            evidence_root
            / "cases"
            / case_id
            / "events"
            / f"support__{_safe_name(event_id)}"
            / "comparison_manifest.json"
        )
        manifest = _read_json(manifest_path)
        arm_cards: list[str] = []
        for arm in ("support_top1", "support_top2", "support_top3"):
            result = arm_rows[arm]
            evidence = (manifest.get("arms") or {})[arm].get("items") or []
            figures = "".join(
                _figure(
                    Path(str(item["path"])),
                    output_path,
                    f"{item.get('view_id')} · {item.get('role')}",
                )
                for item in evidence
            )
            verdict_class = (
                "invalid" if result["predicted_label"] == "invalid" else "valid"
            )
            arm_cards.append(
                f'<article class="arm"><h3>{html.escape(arm)}</h3>'
                f'<p class="verdict {verdict_class}">'
                f'{html.escape(result["predicted_label"])} · '
                f'{html.escape(result["confidence"])}</p>'
                f'<div class="images">{figures}</div>'
                f'<p>{html.escape(result["reason"])}</p></article>'
            )
        cards.append(
            f'<section class="event"><h2>{html.escape(case_id)} · '
            f'{html.escape(event_id)}</h2>'
            f'<p>{html.escape(str(manifest.get("severity_class")))} · '
            f'GT {html.escape(gt_label)}</p>'
            f'<div class="arms">{"".join(arm_cards)}</div></section>'
        )
    subtitle = (
        "8 scored invalid events; green means the invalid GT was detected."
        if gt_label == "invalid"
        else "6 ambiguous events; no verdict is scored as correct. Manual review only."
    )
    return _standalone(
        title,
        f"<header><h1>{html.escape(title)}</h1><p>{subtitle}</p></header>"
        + "".join(cards),
    )


def _dashboard_html(
    valid_summary: list[dict[str, str]],
    topk_summary: list[dict[str, str]],
    extended_summary: list[dict[str, str]],
) -> str:
    valid_metrics = {
        row["group"]: row
        for row in valid_summary
        if row["group_type"] == "metric"
    }
    invalid_metrics = {
        (row["arm"], row["group"]): row
        for row in extended_summary
        if row["group_type"] == "metric"
    }
    topk_overall = [
        row for row in topk_summary
        if row["group_type"] == "gt_label" and row["group"] == "invalid"
    ]
    rows = [
        {
            "metric": metric.title(),
            "valid_specificity": f"{float(valid_metrics[metric]['specificity_all'])*100:.1f}%",
            "global_invalid_recall": (
                f"{float(invalid_metrics[('fixed_global_highlight', metric)]['invalid_recall'])*100:.1f}%"
            ),
            "local_invalid_recall": (
                f"{float(invalid_metrics[('metric_local_highlight', metric)]['invalid_recall'])*100:.1f}%"
            ),
            "interpretation": {
                "collision": "GT / mesh uncertainty dominates remaining error",
                "oob": "v2 fixes clean FP; view policy limits invalid recall",
                "support": "TopK local-angle count is the active variable",
            }[metric],
        }
        for metric in ("collision", "oob", "support")
    ]
    topk_bars = "".join(
        _bar(
            row["arm"].replace("support_", "").upper(),
            float(row["invalid_recall_all"]) * 100.0,
            f"{row['invalid_detected']}/{row['invalid_gt_total']}",
        )
        for row in topk_overall
    )
    body = f"""
<header><h1>Completed VisualConfig calibration</h1>
<p>GPT-5.6-Sol · frozen local evidence · binary GT scored separately from ambiguous</p></header>
<section>{_table(["metric","valid_specificity","global_invalid_recall","local_invalid_recall","interpretation"], rows)}</section>
<section><h2>Support nested local TopK</h2><div class="bars">{topk_bars}</div></section>
<section class="decision">
<h2>Decision state</h2>
<ul>
<li>OOB: oob_p0b_v2 is the current contract; specificity is 50/50 and two-arm invalid recall is 6/8 for both arms.</li>
<li>OOB: local raw, local highlight, and compact local-first each reach 8/8 in the nine-arm replay; global context is not a measured requirement.</li>
<li>Support: K3 local highlight + same global context is the provisional winner.</li>
<li>Collision: retain deterministic local evidence; audit frozen valid/invalid geometry conflicts.</li>
<li>VLM-active camera search remains lower priority.</li>
</ul></section>
"""
    return _standalone("Completed VisualConfig calibration", body)


def _bar(label: str, value: float, count: str) -> str:
    bounded = max(0.0, min(100.0, value))
    return (
        '<div class="bar-row">'
        f'<span>{html.escape(label)}</span>'
        '<div class="track">'
        f'<div class="fill" style="width:{bounded:.3f}%"></div>'
        "</div>"
        f'<b>{value:.1f}%</b><small>{html.escape(count)}</small>'
        "</div>"
    )


def _table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(_display(row.get(column)))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _figure(image_path: Path, output_path: Path, caption: str) -> str:
    source = os.path.relpath(image_path.resolve(), output_path.parent.resolve())
    return (
        "<figure>"
        f'<img loading="lazy" src="{html.escape(Path(source).as_posix())}" '
        f'alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _standalone(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme:light dark; --bg:#f7f7f5; --fg:#171717; --muted:#606060; --card:#fff; --line:#d8d8d3; --series:#3478c7; --good:#217a4b; --bad:#b43c3c; }}
@media(prefers-color-scheme:dark) {{ :root {{ --bg:#121313; --fg:#ededeb; --muted:#aaa; --card:#1b1c1c; --line:#343636; --series:#6da6e5; --good:#5fc68c; --bad:#ef7777; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; }}
header,section {{ max-width:1500px; margin:0 auto 30px; }} h1,h2,h3 {{ font-weight:500; }} h1 {{ margin-bottom:4px; }} h2 {{ margin-top:0; }} p {{ color:var(--muted); }}
.stats {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }} .stats div {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.stats b {{ display:block; font-size:28px; font-weight:500; }} .stats span {{ color:var(--muted); }}
.bars {{ display:grid; gap:12px; max-width:900px; }} .bar-row {{ display:grid; grid-template-columns:90px minmax(120px,1fr) 65px 60px; gap:10px; align-items:center; }}
.track {{ height:18px; background:color-mix(in srgb,var(--line) 55%,transparent); border-radius:4px; overflow:hidden; }} .fill {{ height:100%; background:var(--series); }}
.table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; background:var(--card); }} th,td {{ border-bottom:1px solid var(--line); padding:9px; text-align:left; vertical-align:top; }} th {{ position:sticky; top:0; background:var(--card); font-weight:500; }} td:last-child {{ min-width:280px; }}
.event {{ border-top:1px solid var(--line); padding-top:20px; }} .arms {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} .arm {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }}
.images {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }} figure {{ margin:0; }} img {{ display:block; width:100%; height:auto; border-radius:5px; }} figcaption {{ color:var(--muted); font-size:11px; overflow-wrap:anywhere; }}
.verdict {{ font-weight:500; }} .verdict.invalid {{ color:var(--good); }} .verdict.valid {{ color:var(--bad); }} .decision li {{ margin:8px 0; }}
@media(max-width:900px) {{ .arms,.stats {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:70px 1fr 60px; }} .bar-row small {{ display:none; }} }}
</style></head><body>{body}</body></html>"""


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_hash_inventory(root: Path, path: Path) -> None:
    lines = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.resolve() != path.resolve():
            lines.append(
                f"{_file_sha256(candidate)}  {candidate.relative_to(root).as_posix()}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._") or "event"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
