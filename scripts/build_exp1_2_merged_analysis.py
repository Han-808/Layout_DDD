#!/usr/bin/env python3
"""Build the merged Exp 1.2 analysis from the two completed VisualConfig jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "experiment_analysis"
    / "exp1_1_visual_config_20260723"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "experiment_analysis"
    / "exp1_2_merged_20260723"
)

COMMON_ARMS = {
    "fixed_global_highlight": "fixed_global_highlight",
    "metric_local_highlight": "presence_local_raw_highlight",
}
METRICS = ("collision", "oob", "support")
GROUPS = ("overall", *METRICS)

HUMAN_ANNOTATIONS = (
    (
        1,
        1,
        "fine_edge_easy_001",
        "collision",
        "obj_000|obj_004",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        1,
        2,
        "fine_edge_easy_001",
        "oob",
        "obj_000",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        1,
        3,
        "fine_edge_easy_001",
        "support",
        "obj_002",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        2,
        1,
        "fine_edge_easy_003",
        "collision",
        "obj_000|obj_002",
        "FTFT",
        "",
        "human_reviewed",
    ),
    (
        2,
        2,
        "fine_edge_easy_003",
        "oob",
        "obj_000",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        2,
        3,
        "fine_edge_easy_003",
        "support",
        "obj_001",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        3,
        1,
        "fine_edge_easy_004",
        "collision",
        "obj_000|obj_001",
        "FFTT",
        "",
        "human_reviewed",
    ),
    (
        3,
        2,
        "fine_edge_easy_004",
        "oob",
        "obj_000",
        "FTTT",
        "",
        "human_reviewed",
    ),
    (
        3,
        3,
        "fine_edge_easy_004",
        "support",
        "obj_002",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        4,
        1,
        "fine_edge_easy_007",
        "collision",
        "obj_001|obj_004",
        "FTTT",
        "",
        "human_reviewed",
    ),
    (
        4,
        2,
        "fine_edge_easy_007",
        "oob",
        "obj_000",
        "TTTF",
        "",
        "human_reviewed",
    ),
    (
        4,
        3,
        "fine_edge_easy_007",
        "support",
        "obj_002",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        5,
        1,
        "fine_edge_prompt_010",
        "oob",
        "obj_000",
        "TTTT",
        "太难判断; 四个 calls 均按 correct 计入",
        "uncertain_forced_correct",
    ),
    (
        5,
        2,
        "fine_edge_prompt_010",
        "support",
        "obj_005",
        "TTTT",
        "",
        "human_reviewed",
    ),
    (
        5,
        3,
        "fine_edge_prompt_010",
        "support",
        "obj_006",
        "TTTT",
        "",
        "human_reviewed",
    ),
)

ABLATIONS = (
    (
        "local_vs_global",
        "Local raw vs fixed global raw",
        "fixed_global",
        "presence_local_raw",
        "方向性证据",
        "Local raw is the strongest two-image baseline, but n=24 is too small "
        "for a secure superiority claim.",
    ),
    (
        "highlight_global",
        "Global raw vs global raw + highlight",
        "fixed_global",
        "fixed_global_highlight",
        "可能是随机波动",
        "Duplicating the same global poses as raw and highlight shows no gain; "
        "this does not prove highlight itself is harmful.",
    ),
    (
        "highlight_local",
        "Local raw vs local raw + highlight",
        "presence_local_raw",
        "presence_local_raw_highlight",
        "可能是随机波动",
        "Duplicated highlight uses more images without improving recall in this "
        "pilot; targeted highlight remains plausible.",
    ),
    (
        "global_context_raw",
        "Local raw vs global-top + local raw",
        "presence_local_raw",
        "presence_global_local_raw",
        "方向性证据",
        "A global view is not universally required; the conclusion is stronger "
        "as a non-requirement than as evidence of harm.",
    ),
    (
        "global_context_highlight",
        "Local highlight vs full global-first",
        "presence_local_raw_highlight",
        "deterministic_metric_local",
        "无可分辨增益",
        "The two packets tie overall and exchange one discordant event each.",
    ),
    (
        "image_order_full",
        "Full global-first vs full local-first",
        "deterministic_metric_local",
        "order_local_first_full",
        "可能是随机波动",
        "A one-event difference cannot establish image-order sensitivity.",
    ),
    (
        "budget_global_first",
        "Global-first budget 5 vs budget 3",
        "deterministic_metric_local",
        "budget_global_first_compact",
        "可能是随机波动",
        "Compact global-first loses one event overall; no general budget law "
        "follows.",
    ),
    (
        "budget_local_first",
        "Local-first budget 5 vs budget 3",
        "order_local_first_full",
        "budget_local_first_compact",
        "Metric-specific signal",
        "The overall change is one event, but Support loses two while OOB gains "
        "one, motivating per-metric VisualConfig.",
    ),
)

ARM_LABELS = {
    "fixed_global": "Global raw",
    "fixed_global_highlight": "Global raw + highlight",
    "presence_local_raw": "Local raw",
    "presence_local_raw_highlight": "Local raw + highlight",
    "presence_global_local_raw": "Global top + local raw",
    "deterministic_metric_local": "Full global-first",
    "order_local_first_full": "Full local-first",
    "budget_global_first_compact": "Compact global-first",
    "budget_local_first_compact": "Compact local-first",
}


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    job1_rows = _read_tsv(source_root / "data" / "job1" / "per_event.tsv")
    job2_rows = _read_tsv(source_root / "data" / "job2" / "per_event.tsv")
    _validate_inputs(job1_rows, job2_rows)

    human_rows = _build_human_rows(job1_rows)
    human_call_rows = _human_call_rows(human_rows)
    human_summary = _human_summary(human_call_rows)
    human_call_summary = _human_call_summary(human_call_rows)
    human_local_vs_global = _human_local_vs_global(human_call_rows)
    repeatability = _repeatability_rows(job1_rows, job2_rows)
    ablations = _ablation_rows(job2_rows)
    arm_matrix = _arm_metric_matrix(job2_rows)
    merged = _merged_judgements(job1_rows, job2_rows)
    takeaways = _takeaway_rows(
        human_summary,
        human_local_vs_global,
        repeatability,
        ablations,
    )

    _write_tsv(data_root / "human_ambiguous_audit.tsv", human_rows)
    _write_tsv(data_root / "human_accuracy_summary.tsv", human_summary)
    _write_tsv(data_root / "human_call_accuracy.tsv", human_call_rows)
    _write_tsv(
        data_root / "human_call_accuracy_summary.tsv",
        human_call_summary,
    )
    _write_tsv(
        data_root / "human_local_vs_global.tsv",
        human_local_vs_global,
    )
    _write_tsv(data_root / "repeatability.tsv", repeatability)
    _write_tsv(data_root / "ablation_overall_metric.tsv", ablations)
    _write_tsv(data_root / "arm_metric_matrix.tsv", arm_matrix)
    _write_tsv(data_root / "merged_judgements.tsv", merged)
    _write_tsv(data_root / "takeaways_with_confidence.tsv", takeaways)

    summary = {
        "schema_version": "exp1_2_merged_visual_config_v2",
        "source_jobs": {
            "job1_rows": len(job1_rows),
            "job2_rows": len(job2_rows),
            "total_resolved_judgements": len(job1_rows) + len(job2_rows),
        },
        "unique_events": {
            "constructed_invalid": 24,
            "human_audited_ambiguous": 15,
            "total": 39,
        },
        "human_audit": {
            "events": len(human_rows),
            "calls": len(human_call_rows),
            "correct_calls": sum(
                int(row["human_correct"]) for row in human_call_rows
            ),
            "uncertain_forced_correct_events": sum(
                row["human_audit_confidence"] == "uncertain_forced_correct"
                for row in human_rows
            ),
            "label_semantics": (
                "direct per-call human correctness; no event-level GT inferred"
            ),
        },
        "human_call_accuracy": next(
            row
            for row in human_call_summary
            if row["arm"] == "all_calls" and row["group"] == "overall"
        ),
        "human_local_vs_global": next(
            row
            for row in human_local_vs_global
            if row["group"] == "overall"
        ),
        "exact_input_repeatability": {
            row["stratum"]: row
            for row in repeatability
            if row["arm"] == "all_common_arms"
            and row["group"] == "overall"
        },
        "ablation_count": len(ABLATIONS),
        "unresolved_count": sum(
            row.get("resolved") != "1" for row in merged
        ),
    }
    _write_json(data_root / "summary.json", summary)
    print(json.dumps({
        "output_root": str(output_root),
        "human_accuracy": (
            f"{summary['human_call_accuracy']['correct']}/"
            f"{summary['human_call_accuracy']['total']}"
        ),
        "ablation_count": len(ABLATIONS),
        "merged_judgements": len(merged),
        "unresolved": summary["unresolved_count"],
    }, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _validate_inputs(
    job1_rows: list[dict[str, str]],
    job2_rows: list[dict[str, str]],
) -> None:
    if len(job1_rows) != 108:
        raise ValueError(f"expected 108 Job 1 rows, found {len(job1_rows)}")
    if len(job2_rows) != 216:
        raise ValueError(f"expected 216 Job 2 rows, found {len(job2_rows)}")
    unresolved = [
        row for row in (*job1_rows, *job2_rows) if row.get("resolved") != "1"
    ]
    if unresolved:
        raise ValueError(f"found {len(unresolved)} unresolved judgements")


def _build_human_rows(
    job1_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    predictions: dict[
        tuple[str, str, str],
        dict[tuple[str, int], str],
    ] = defaultdict(dict)
    for row in job1_rows:
        if row["stratum"] != "ambiguous":
            continue
        key = (row["case_id"], row["metric"], row["event_id"])
        predictions[key][(row["arm"], int(row["repeat"]))] = row[
            "predicted_label"
        ]

    expected = {
        (case_id, metric, event_id)
        for _, _, case_id, metric, event_id, _, _, _ in HUMAN_ANNOTATIONS
    }
    if set(predictions) != expected:
        raise RuntimeError("human annotation keys do not match ambiguous events")

    rows: list[dict[str, Any]] = []
    for (
        scene_index,
        event_index,
        case_id,
        metric,
        event_id,
        human_call_pattern,
        note,
        audit_confidence,
    ) in HUMAN_ANNOTATIONS:
        if len(human_call_pattern) != 4 or set(human_call_pattern) - {"T", "F"}:
            raise RuntimeError(
                f"invalid human call pattern for {case_id}/{metric}/{event_id}: "
                f"{human_call_pattern!r}"
            )
        values = predictions[(case_id, metric, event_id)]
        ordered = (
            values[("fixed_global_highlight", 1)],
            values[("fixed_global_highlight", 2)],
            values[("metric_local_highlight", 1)],
            values[("metric_local_highlight", 2)],
        )
        if len(values) != 4:
            raise RuntimeError(
                f"expected four predictions for {case_id}/{metric}/{event_id}"
            )
        labels = list(values.values())
        counts = Counter(labels)
        if counts["valid"] == counts["invalid"]:
            consensus = "split"
        else:
            consensus = counts.most_common(1)[0][0]
        rows.append({
            "scene_index": scene_index,
            "event_index": event_index,
            "case_id": case_id,
            "metric": metric,
            "event_id": event_id,
            "human_call_pattern": human_call_pattern,
            "human_correct_calls": human_call_pattern.count("T"),
            "human_all_calls_correct": int(human_call_pattern == "TTTT"),
            "human_audit_note": note,
            "human_audit_confidence": audit_confidence,
            "annotation_source": "user_redelivery_2026-07-23",
            "fixed_repeat_1": ordered[0],
            "fixed_repeat_2": ordered[1],
            "local_repeat_1": ordered[2],
            "local_repeat_2": ordered[3],
            "fixed_repeat_1_human": human_call_pattern[0],
            "fixed_repeat_2_human": human_call_pattern[1],
            "local_repeat_1_human": human_call_pattern[2],
            "local_repeat_2_human": human_call_pattern[3],
            "four_call_consensus": consensus,
        })
    return rows


def _human_summary(
    human_call_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        selected = [
            row for row in human_call_rows
            if group == "overall" or row["metric"] == group
        ]
        correct = sum(int(row["human_correct"]) for row in selected)
        low, high = _wilson(correct, len(selected))
        rows.append({
            "group": group,
            "correct": correct,
            "total": len(selected),
            "accuracy": f"{correct / len(selected):.4f}",
            "wilson_95_low": f"{low:.4f}",
            "wilson_95_high": f"{high:.4f}",
        })
    return rows


def _human_call_rows(
    human_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    prediction_fields = (
        (
            "fixed_global_highlight",
            1,
            "fixed_repeat_1",
            "fixed_repeat_1_human",
        ),
        (
            "fixed_global_highlight",
            2,
            "fixed_repeat_2",
            "fixed_repeat_2_human",
        ),
        (
            "metric_local_highlight",
            1,
            "local_repeat_1",
            "local_repeat_1_human",
        ),
        (
            "metric_local_highlight",
            2,
            "local_repeat_2",
            "local_repeat_2_human",
        ),
    )
    for row in human_rows:
        for arm, repeat, field, human_field in prediction_fields:
            predicted = row[field]
            records.append({
                "scene_index": row["scene_index"],
                "event_index": row["event_index"],
                "case_id": row["case_id"],
                "metric": row["metric"],
                "event_id": row["event_id"],
                "arm": arm,
                "repeat": repeat,
                "predicted_label": predicted,
                "human_correct": int(row[human_field] == "T"),
                "human_symbol": row[human_field],
                "human_audit_note": row["human_audit_note"],
                "human_audit_confidence": row["human_audit_confidence"],
                "annotation_source": row["annotation_source"],
            })
    return records


def _human_call_summary(
    human_call_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in (
        "all_calls",
        "fixed_global_highlight",
        "metric_local_highlight",
    ):
        for group in GROUPS:
            selected = [
                row
                for row in human_call_rows
                if (arm == "all_calls" or row["arm"] == arm)
                and (group == "overall" or row["metric"] == group)
            ]
            correct = sum(int(row["human_correct"]) for row in selected)
            low, high = _wilson(correct, len(selected))
            records.append({
                "arm": arm,
                "group": group,
                "correct": correct,
                "total": len(selected),
                "accuracy": f"{correct / len(selected):.4f}",
                "wilson_95_low": f"{low:.4f}",
                "wilson_95_high": f"{high:.4f}",
            })
    return records


def _human_local_vs_global(
    human_call_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["arm"],
            row["case_id"],
            row["metric"],
            row["event_id"],
            int(row["repeat"]),
        ): row
        for row in human_call_rows
    }
    records: list[dict[str, Any]] = []
    for group in GROUPS:
        keys = {
            (row["case_id"], row["metric"], row["event_id"], int(row["repeat"]))
            for row in human_call_rows
            if row["arm"] == "fixed_global_highlight"
            and (group == "overall" or row["metric"] == group)
        }
        global_rows = [
            by_key[("fixed_global_highlight", *key)] for key in keys
        ]
        local_rows = [
            by_key[("metric_local_highlight", *key)] for key in keys
        ]
        global_correct = sum(int(row["human_correct"]) for row in global_rows)
        local_correct = sum(int(row["human_correct"]) for row in local_rows)
        both_correct = sum(
            int(left["human_correct"]) and int(right["human_correct"])
            for left, right in zip(global_rows, local_rows)
        )
        both_wrong = sum(
            not int(left["human_correct"]) and not int(right["human_correct"])
            for left, right in zip(global_rows, local_rows)
        )
        global_only = sum(
            int(left["human_correct"]) and not int(right["human_correct"])
            for left, right in zip(global_rows, local_rows)
        )
        local_only = sum(
            not int(left["human_correct"]) and int(right["human_correct"])
            for left, right in zip(global_rows, local_rows)
        )
        records.append({
            "group": group,
            "paired_calls": len(keys),
            "global_correct": global_correct,
            "local_correct": local_correct,
            "global_accuracy": f"{global_correct / len(keys):.4f}",
            "local_accuracy": f"{local_correct / len(keys):.4f}",
            "local_minus_global_pp": (
                f"{(local_correct - global_correct) / len(keys) * 100:.2f}"
            ),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "global_only": global_only,
            "local_only": local_only,
            "mcnemar_exact_p": f"{_mcnemar_exact(global_only, local_only):.4f}",
        })
    return records


def _repeatability_rows(
    job1_rows: list[dict[str, str]],
    job2_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stratum in ("invalid", "ambiguous", "pooled"):
        for arm in (*COMMON_ARMS, "all_common_arms"):
            for group in GROUPS:
                pairs: list[tuple[str, str]] = []
                selected_arms = (
                    tuple(COMMON_ARMS)
                    if arm == "all_common_arms"
                    else (arm,)
                )
                for selected_arm in selected_arms:
                    if stratum in ("invalid", "pooled"):
                        job2_arm = COMMON_ARMS[selected_arm]
                        left = {
                            (row["case_id"], row["metric"], row["event_id"]):
                            row["predicted_label"]
                            for row in job1_rows
                            if row["stratum"] == "invalid"
                            and row["arm"] == selected_arm
                            and (group == "overall" or row["metric"] == group)
                        }
                        right = {
                            (row["case_id"], row["metric"], row["event_id"]):
                            row["predicted_label"]
                            for row in job2_rows
                            if row["arm"] == job2_arm
                            and (group == "overall" or row["metric"] == group)
                        }
                        if set(left) != set(right):
                            raise RuntimeError(
                                f"invalid repeat key mismatch: "
                                f"{selected_arm} {group}"
                            )
                        pairs.extend((left[key], right[key]) for key in left)
                    if stratum in ("ambiguous", "pooled"):
                        by_repeat: dict[
                            int,
                            dict[tuple[str, str, str], str],
                        ] = {}
                        for repeat in (1, 2):
                            by_repeat[repeat] = {
                                (
                                    row["case_id"],
                                    row["metric"],
                                    row["event_id"],
                                ): row["predicted_label"]
                                for row in job1_rows
                                if row["stratum"] == "ambiguous"
                                and row["arm"] == selected_arm
                                and int(row["repeat"]) == repeat
                                and (
                                    group == "overall"
                                    or row["metric"] == group
                                )
                            }
                        if set(by_repeat[1]) != set(by_repeat[2]):
                            raise RuntimeError(
                                f"ambiguous repeat key mismatch: "
                                f"{selected_arm} {group}"
                            )
                        pairs.extend(
                            (by_repeat[1][key], by_repeat[2][key])
                            for key in by_repeat[1]
                        )
                agreement = sum(left == right for left, right in pairs)
                low, high = _wilson(agreement, len(pairs))
                records.append({
                    "stratum": stratum,
                    "arm": arm,
                    "group": group,
                    "agreements": agreement,
                    "pairs": len(pairs),
                    "agreement_rate": f"{agreement / len(pairs):.4f}",
                    "wilson_95_low": f"{low:.4f}",
                    "wilson_95_high": f"{high:.4f}",
                })
    return records


def _ablation_rows(
    job2_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["arm"], row["case_id"], row["metric"], row["event_id"]): row
        for row in job2_rows
    }
    records: list[dict[str, Any]] = []
    for (
        question,
        label,
        left_arm,
        right_arm,
        confidence,
        interpretation,
    ) in ABLATIONS:
        for group in GROUPS:
            left = {
                (case_id, metric, event_id): row
                for (arm, case_id, metric, event_id), row in by_key.items()
                if arm == left_arm and (group == "overall" or metric == group)
            }
            right = {
                (case_id, metric, event_id): row
                for (arm, case_id, metric, event_id), row in by_key.items()
                if arm == right_arm and (group == "overall" or metric == group)
            }
            if set(left) != set(right):
                raise RuntimeError(f"ablation keys differ: {question} {group}")
            left_correct = sum(row["match"] == "1" for row in left.values())
            right_correct = sum(row["match"] == "1" for row in right.values())
            left_only = sum(
                left[key]["match"] == "1" and right[key]["match"] != "1"
                for key in left
            )
            right_only = sum(
                left[key]["match"] != "1" and right[key]["match"] == "1"
                for key in left
            )
            total = len(left)
            records.append({
                "question": question,
                "label": label,
                "group": group,
                "left_arm": left_arm,
                "left_label": ARM_LABELS[left_arm],
                "right_arm": right_arm,
                "right_label": ARM_LABELS[right_arm],
                "total": total,
                "left_correct": left_correct,
                "right_correct": right_correct,
                "left_accuracy": f"{left_correct / total:.4f}",
                "right_accuracy": f"{right_correct / total:.4f}",
                "right_minus_left_pp": (
                    f"{(right_correct - left_correct) / total * 100:.2f}"
                ),
                "paired_left_only": left_only,
                "paired_right_only": right_only,
                "mcnemar_exact_p": f"{_mcnemar_exact(left_only, right_only):.4f}",
                "confidence": confidence,
                "interpretation": interpretation,
            })
    return records


def _arm_metric_matrix(
    job2_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in ARM_LABELS:
        selected = [row for row in job2_rows if row["arm"] == arm]
        if not selected:
            raise RuntimeError(f"missing arm: {arm}")
        record: dict[str, Any] = {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "image_budget": selected[0]["image_budget"],
        }
        for group in GROUPS:
            group_rows = [
                row for row in selected
                if group == "overall" or row["metric"] == group
            ]
            correct = sum(row["match"] == "1" for row in group_rows)
            record[f"{group}_correct"] = correct
            record[f"{group}_total"] = len(group_rows)
            record[f"{group}_accuracy"] = f"{correct / len(group_rows):.4f}"
        records.append(record)
    return records


def _merged_judgements(
    job1_rows: list[dict[str, str]],
    job2_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for experiment, rows in (("job1_two_arm", job1_rows), ("job2_nine_arm", job2_rows)):
        for row in rows:
            record: dict[str, Any] = {"experiment": experiment}
            record.update(row)
            records.append(record)
    return records


def _takeaway_rows(
    human_summary: list[dict[str, Any]],
    human_local_vs_global: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    human = next(row for row in human_summary if row["group"] == "overall")
    human_pair = next(
        row for row in human_local_vs_global if row["group"] == "overall"
    )
    pooled = next(
        row for row in repeatability
        if row["stratum"] == "pooled"
        and row["arm"] == "all_common_arms"
        and row["group"] == "overall"
    )
    overall_ablations = {
        row["question"]: row for row in ablations if row["group"] == "overall"
    }
    return [
        {
            "takeaway": "Human-reviewed call accuracy is high",
            "confidence": "方向较清楚, 但样本与人工不确定性仍限制结论",
            "backing_data": (
                f"{human['correct']}/{human['total']} = "
                f"{float(human['accuracy'])*100:.1f}%; "
                f"Wilson 95% CI "
                f"{float(human['wilson_95_low'])*100:.1f}–"
                f"{float(human['wilson_95_high'])*100:.1f}%"
            ),
            "boundary": (
                "Labels are direct per-call human correctness, not an inferred "
                "event GT. One hard OOB event was explicitly counted TTTT by "
                "the reviewer because it was too difficult to resolve."
            ),
        },
        {
            "takeaway": "Exact-input judgement is stable but not deterministic",
            "confidence": "较有把握",
            "backing_data": (
                f"{pooled['agreements']}/{pooled['pairs']} = "
                f"{float(pooled['agreement_rate'])*100:.1f}%"
            ),
            "boundary": "One residual OOB near-wall interpretation is systematic.",
        },
        {
            "takeaway": "Deterministic local is a sufficient default camera baseline",
            "confidence": "方向性 pilot evidence; 未达到统计显著",
            "backing_data": (
                _ablation_evidence(overall_ablations["local_vs_global"])
                + "; direct human edge labels: global "
                f"{human_pair['global_correct']}/{human_pair['paired_calls']} "
                "vs local "
                f"{human_pair['local_correct']}/{human_pair['paired_calls']}; "
                f"exact p={human_pair['mcnemar_exact_p']}"
            ),
            "boundary": (
                "The exact test treats repeat calls as paired rows; only five "
                "independent scenes remain, and one event is human-uncertain."
            ),
        },
        {
            "takeaway": "Do not duplicate every pose as raw + highlight",
            "confidence": "工程结论较有把握; accuracy effect不确定",
            "backing_data": (
                _ablation_evidence(overall_ablations["highlight_global"])
                + "; "
                + _ablation_evidence(overall_ablations["highlight_local"])
            ),
            "boundary": "Targeted Collision highlight remains reasonable.",
        },
        {
            "takeaway": "Global context is optional, not universally required",
            "confidence": "方向性证据",
            "backing_data": (
                _ablation_evidence(overall_ablations["global_context_raw"])
                + "; highlighted comparison ties"
            ),
            "boundary": "A global top view can still be retained for selected OOB cases.",
        },
        {
            "takeaway": "Image order is not a current priority",
            "confidence": "可能是随机波动",
            "backing_data": _ablation_evidence(
                overall_ablations["image_order_full"]
            ),
            "boundary": "Only one of 24 decisions changes in net accuracy.",
        },
        {
            "takeaway": "Support should spend budget on distinct local angles",
            "confidence": "Metric-specific signal",
            "backing_data": (
                "Full local-first Support 7/8 vs compact local-first 5/8; "
                "auxiliary TopK pilot reaches 8/8 at Top3."
            ),
            "boundary": "Exact Top3 production precision still needs matched valid cases.",
        },
        {
            "takeaway": "Per-metric VisualConfig is justified",
            "confidence": "较有把握",
            "backing_data": (
                "OOB reaches 8/8 with local raw; Support peaks at 7/8 with full "
                "local-first; Collision remains 5/8 across local variants."
            ),
            "boundary": "The exact policy is pilot-level, not publication-level frozen.",
        },
    ]


def _ablation_evidence(row: dict[str, Any]) -> str:
    return (
        f"{row['left_label']} {row['left_correct']}/{row['total']} vs "
        f"{row['right_label']} {row['right_correct']}/{row['total']}; "
        f"Δ {float(row['right_minus_left_pp']):+.1f} pp; "
        f"exact p={row['mcnemar_exact_p']}"
    )


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (
        proportion + z * z / (2 * total)
    ) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
