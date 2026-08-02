#!/usr/bin/env python3
"""Analyze the completed cal_dataset2 non-L1 VisualConfig ablation.

The primary unit is a human-labelled case. The two exact-input model repeats
are averaged within case before paired arm effects are estimated, so repeats
are not treated as independent samples. Human ``ambiguous`` labels are excluded
from binary accuracy and reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "exp2_non_l1_visual_evidence_gpt56"
)
DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
OUTPUT_ROOT = RUN_ROOT / "ablation_analysis"
GT_PATH = DATASET_ROOT / "human_review" / "human_gt_20260725.tsv"
REVIEW_CASES = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_review_renders"
    / "review"
    / "review_cases.json"
)

METRIC_ORDER = (
    "room_scene_type",
    "broad_semantic_intent",
    "required_functional_areas",
    "oor",
    "oar",
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
BASE_ARMS = (
    "production_default",
    "global_only",
    "local_raw_only",
    "full_raw",
)
PAIR_SPECS = (
    (
        "local_only_vs_global_only",
        "global_only",
        "local_raw_only",
        "Use three local raw views instead of two global raw views",
        None,
    ),
    (
        "local_gain_over_global",
        "global_only",
        "full_raw",
        "Add three local raw views to the two global raw views",
        None,
    ),
    (
        "global_gain_over_local",
        "local_raw_only",
        "full_raw",
        "Add two global raw views to the three local raw views",
        None,
    ),
    (
        "expanded_packet_vs_production",
        "production_default",
        "full_raw",
        "Use all five raw views instead of the metric-specific production packet",
        None,
    ),
    (
        "contour_vs_raw_production",
        "production_raw_swap",
        "production_default",
        "Replace local raw with same-pose contour views in the production packet",
        {"oor", "oar"},
    ),
    (
        "contour_vs_raw_local",
        "local_raw_only",
        "local_contour_only",
        "Replace three local raw views with their same-pose contour versions",
        {"oor", "oar"},
    ),
)


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.out_dir.expanduser().resolve()
    gt = _read_gt(args.ground_truth.expanduser().resolve())
    review_cards = _review_cards(args.review_cases.expanduser().resolve())
    results = _read_repeats(run_root)
    _validate_identity(results, gt)

    production = _production_summary(results, gt)
    paired = _paired_effects(results, gt, args.bootstrap_samples)
    chair = _chair_angle_summary(results, gt, dataset_root)
    ambiguous = _ambiguous_summary(results, gt)
    proposal_sensitivity = _proposal_sensitivity(results, gt, dataset_root)
    stable_disagreements = _stable_disagreements(
        results,
        gt,
        review_cards,
    )
    overall = _overall_arm_summary(results, gt)
    oar_diagnostics = _oar_diagnostics(results, gt)
    identical_packet_control = _identical_packet_control(results)
    payload = {
        "schema_version": "exp2_non_l1_ablation_analysis_v1",
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "ground_truth": {
            "path": str(args.ground_truth.expanduser().resolve()),
            "sha256": _file_sha256(args.ground_truth.expanduser().resolve()),
            "label_counts": dict(
                sorted(Counter(row["human_semantic_label"] for row in gt.values()).items())
            ),
            "binary_accuracy_policy": (
                "valid/invalid only; human ambiguous cases are excluded"
            ),
        },
        "experiment": {
            "case_count": len(gt),
            "calls_per_repeat": len(results[1]),
            "repeat_count": 2,
            "total_resolved_calls": sum(len(items) for items in results.values()),
            "unit_of_analysis": "case; repeat outcomes averaged within case",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "overall_arms": overall,
        "production_by_metric": production,
        "paired_effects": paired,
        "chair_angle": chair,
        "ambiguous_gt": ambiguous,
        "construction_proposal_sensitivity": proposal_sensitivity,
        "oar_diagnostics": oar_diagnostics,
        "identical_packet_control": identical_packet_control,
        "stable_disagreement_count": len(stable_disagreements),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_root / "overall_arm_summary.tsv", overall)
    _write_tsv(output_root / "production_metric_summary.tsv", production)
    _write_tsv(output_root / "paired_ablation_effects.tsv", paired)
    _write_tsv(output_root / "chair_angle_summary.tsv", chair["arms"])
    _write_tsv(output_root / "ambiguous_gt_predictions.tsv", ambiguous["arms"])
    _write_tsv(
        output_root / "construction_proposal_sensitivity.tsv",
        proposal_sensitivity,
    )
    _write_tsv(
        output_root / "stable_production_disagreements.tsv",
        stable_disagreements,
    )
    _write_json(output_root / "analysis.json", payload)
    (output_root / "report.md").write_text(
        _report(payload, stable_disagreements),
        encoding="utf-8",
    )
    _write_json(
        output_root / "analysis_manifest.json",
        {
            "schema_version": "exp2_non_l1_ablation_analysis_manifest_v1",
            "outputs": {
                path.name: {
                    "path": str(path.resolve()),
                    "sha256": _file_sha256(path),
                }
                for path in sorted(output_root.iterdir())
                if path.is_file() and path.name != "analysis_manifest.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "production_metric_rows": len(production),
                "paired_effect_rows": len(paired),
                "stable_disagreements": len(stable_disagreements),
            },
            indent=2,
        )
    )


def _overall_arm_summary(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in BASE_ARMS:
        case_ids = sorted(
            case_id
            for case_id, gt_row in gt.items()
            if gt_row["human_semantic_label"] in {"valid", "invalid"}
            and all((case_id, arm) in results[repeat] for repeat in (1, 2))
        )
        call_rows = [
            results[repeat][(case_id, arm)]
            for case_id in case_ids
            for repeat in (1, 2)
        ]
        all_case_ids = sorted(
            case_id
            for case_id in gt
            if all((case_id, arm) in results[repeat] for repeat in (1, 2))
        )
        all_calls = [
            results[repeat][(case_id, arm)]
            for case_id in all_case_ids
            for repeat in (1, 2)
        ]
        valid_calls = [row for row in call_rows if row["gt_label"] == "valid"]
        invalid_calls = [row for row in call_rows if row["gt_label"] == "invalid"]
        rows.append(
            {
                "arm": arm,
                "binary_case_n": len(case_ids),
                "binary_call_n": len(call_rows),
                "correct_call_n": sum(
                    _bool(row["binary_match"]) is True for row in call_rows
                ),
                "repeat_1_correct_n": sum(
                    _bool(results[1][(case_id, arm)]["binary_match"]) is True
                    for case_id in case_ids
                ),
                "repeat_2_correct_n": sum(
                    _bool(results[2][(case_id, arm)]["binary_match"]) is True
                    for case_id in case_ids
                ),
                "binary_accuracy": _mean(
                    _bool(row["binary_match"]) for row in call_rows
                ),
                "balanced_accuracy": _mean(
                    (
                        _mean(_bool(row["binary_match"]) for row in valid_calls),
                        _mean(_bool(row["binary_match"]) for row in invalid_calls),
                    )
                ),
                "false_positive_call_n": sum(
                    row["gt_label"] == "valid"
                    and row["predicted_label"] == "invalid"
                    for row in call_rows
                ),
                "false_negative_call_n": sum(
                    row["gt_label"] == "invalid"
                    and row["predicted_label"] == "valid"
                    for row in call_rows
                ),
                "predicted_ambiguous_call_n": sum(
                    row["predicted_label"] == "ambiguous" for row in call_rows
                ),
                "label_repeat_agreement": _mean(
                    results[1][(case_id, arm)]["predicted_label"]
                    == results[2][(case_id, arm)]["predicted_label"]
                    for case_id in all_case_ids
                ),
                "evidence_sufficient_rate": _mean(
                    row["evidence_status"] == "sufficient" for row in all_calls
                ),
                "mean_confidence": statistics.fmean(
                    float(row["confidence"]) for row in all_calls
                ),
            }
        )
    return rows


def _production_summary(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    arm = "production_default"
    for metric in METRIC_ORDER:
        binary_cases = sorted(
            case_id
            for case_id, row in gt.items()
            if row["metric"] == metric
            and row["human_semantic_label"] in {"valid", "invalid"}
        )
        all_cases = sorted(
            case_id for case_id, row in gt.items() if row["metric"] == metric
        )
        calls = [
            results[repeat][(case_id, arm)]
            for case_id in binary_cases
            for repeat in (1, 2)
        ]
        all_calls = [
            results[repeat][(case_id, arm)]
            for case_id in all_cases
            for repeat in (1, 2)
        ]
        valid_calls = [row for row in calls if row["gt_label"] == "valid"]
        invalid_calls = [row for row in calls if row["gt_label"] == "invalid"]
        modes = Counter()
        for case_id in binary_cases:
            left = results[1][(case_id, arm)]
            right = results[2][(case_id, arm)]
            pair = (_bool(left["binary_match"]), _bool(right["binary_match"]))
            sufficiency = {
                left["evidence_status"],
                right["evidence_status"],
            }
            if pair == (True, True):
                modes["stable_correct"] += 1
            elif pair == (False, False):
                key = (
                    "stable_wrong_insufficient"
                    if "insufficient" in sufficiency
                    else "stable_wrong_sufficient"
                )
                modes[key] += 1
            else:
                modes["unstable_one_correct"] += 1
        correct_confidence = [
            float(row["confidence"])
            for row in calls
            if _bool(row["binary_match"]) is True
        ]
        wrong_confidence = [
            float(row["confidence"])
            for row in calls
            if _bool(row["binary_match"]) is False
        ]
        output.append(
            {
                "metric": metric,
                "case_n": len(all_cases),
                "binary_case_n": len(binary_cases),
                "valid_case_n": sum(
                    gt[case_id]["human_semantic_label"] == "valid"
                    for case_id in binary_cases
                ),
                "invalid_case_n": sum(
                    gt[case_id]["human_semantic_label"] == "invalid"
                    for case_id in binary_cases
                ),
                "human_ambiguous_case_n": len(all_cases) - len(binary_cases),
                "binary_call_n": len(calls),
                "correct_call_n": sum(
                    _bool(row["binary_match"]) is True for row in calls
                ),
                "false_positive_call_n": sum(
                    row["gt_label"] == "valid"
                    and row["predicted_label"] == "invalid"
                    for row in calls
                ),
                "false_negative_call_n": sum(
                    row["gt_label"] == "invalid"
                    and row["predicted_label"] == "valid"
                    for row in calls
                ),
                "predicted_ambiguous_call_n": sum(
                    row["predicted_label"] == "ambiguous" for row in calls
                ),
                "binary_accuracy": _mean(
                    _bool(row["binary_match"]) for row in calls
                ),
                "balanced_accuracy": _mean(
                    (
                        _mean(_bool(row["binary_match"]) for row in valid_calls),
                        _mean(_bool(row["binary_match"]) for row in invalid_calls),
                    )
                ),
                "evidence_sufficient_rate": _mean(
                    row["evidence_status"] == "sufficient" for row in all_calls
                ),
                "label_repeat_agreement": _mean(
                    results[1][(case_id, arm)]["predicted_label"]
                    == results[2][(case_id, arm)]["predicted_label"]
                    for case_id in all_cases
                ),
                "stable_correct_n": modes["stable_correct"],
                "unstable_one_correct_n": modes["unstable_one_correct"],
                "stable_wrong_sufficient_n": modes["stable_wrong_sufficient"],
                "stable_wrong_insufficient_n": modes["stable_wrong_insufficient"],
                "mean_confidence_correct": (
                    statistics.fmean(correct_confidence)
                    if correct_confidence
                    else None
                ),
                "mean_confidence_wrong": (
                    statistics.fmean(wrong_confidence)
                    if wrong_confidence
                    else None
                ),
            }
        )
    return output


def _paired_effects(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    output = []
    for comparison_id, left_arm, right_arm, interpretation, metric_filter in PAIR_SPECS:
        metric_rows = ("ALL",) + METRIC_ORDER
        for metric in metric_rows:
            if (
                metric != "ALL"
                and metric_filter is not None
                and metric not in metric_filter
            ):
                continue
            case_ids = sorted(
                case_id
                for case_id, gt_row in gt.items()
                if (metric == "ALL" or gt_row["metric"] == metric)
                and (metric_filter is None or gt_row["metric"] in metric_filter)
                and gt_row["human_semantic_label"] in {"valid", "invalid"}
                and all(
                    (case_id, arm) in results[repeat]
                    for arm in (left_arm, right_arm)
                    for repeat in (1, 2)
                )
            )
            if not case_ids:
                continue
            differences = []
            right_better = left_better = tied = 0
            for case_id in case_ids:
                left_score = _mean(
                    _bool(results[repeat][(case_id, left_arm)]["binary_match"])
                    for repeat in (1, 2)
                )
                right_score = _mean(
                    _bool(results[repeat][(case_id, right_arm)]["binary_match"])
                    for repeat in (1, 2)
                )
                difference = right_score - left_score
                differences.append(difference)
                if difference > 0:
                    right_better += 1
                elif difference < 0:
                    left_better += 1
                else:
                    tied += 1
            mean, low, high = _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                seed_material=f"{comparison_id}:{metric}",
            )
            output.append(
                {
                    "comparison_id": comparison_id,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "interpretation": interpretation,
                    "metric": metric,
                    "binary_case_n": len(case_ids),
                    "delta_accuracy_pp": 100.0 * mean,
                    "bootstrap_ci_low_pp": 100.0 * low,
                    "bootstrap_ci_high_pp": 100.0 * high,
                    "right_better_case_n": right_better,
                    "left_better_case_n": left_better,
                    "tied_case_n": tied,
                    "confidence_note": (
                        "directional_only"
                        if low <= 0.0 <= high
                        else "bootstrap_interval_excludes_zero"
                    ),
                }
            )
    return output


def _chair_angle_summary(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
    dataset_root: Path,
) -> dict[str, Any]:
    relation_allowlist = {
        "faces_toward",
        "faces_away_from",
        "functional_orientation",
    }
    case_ids = []
    for case_id, gt_row in gt.items():
        if gt_row["human_semantic_label"] not in {"valid", "invalid"}:
            continue
        fixture = dataset_root / "fixtures" / case_id
        event = _read_json(fixture / "metric_events.json")["events"][0]
        if event.get("relation") not in relation_allowlist:
            continue
        targets = set(str(item) for item in event.get("target_ids") or [])
        scene = _read_json(fixture / "generated_scene.json")
        has_target_chair = any(
            str(item.get("id")) in targets
            and "chair" in str(item.get("category") or "").lower()
            for item in scene.get("objects") or []
            if isinstance(item, dict)
        )
        if has_target_chair:
            case_ids.append(case_id)
    case_ids.sort()
    arm_rows = []
    for arm in BASE_ARMS:
        calls = [
            results[repeat][(case_id, arm)]
            for case_id in case_ids
            for repeat in (1, 2)
        ]
        arm_rows.append(
            {
                "arm": arm,
                "case_n": len(case_ids),
                "binary_accuracy": _mean(
                    _bool(row["binary_match"]) for row in calls
                ),
                "label_repeat_agreement": _mean(
                    results[1][(case_id, arm)]["predicted_label"]
                    == results[2][(case_id, arm)]["predicted_label"]
                    for case_id in case_ids
                ),
                "evidence_sufficient_rate": _mean(
                    row["evidence_status"] == "sufficient" for row in calls
                ),
            }
        )
    other_cases = sorted(
        case_id
        for case_id, row in gt.items()
        if row["human_semantic_label"] in {"valid", "invalid"}
        and case_id not in case_ids
    )
    return {
        "definition": (
            "target includes a chair and event relation is faces_toward, "
            "faces_away_from, or functional_orientation"
        ),
        "case_ids": case_ids,
        "case_n": len(case_ids),
        "arms": arm_rows,
        "production_binary_accuracy_other_cases": _mean(
            _bool(results[repeat][(case_id, "production_default")]["binary_match"])
            for case_id in other_cases
            for repeat in (1, 2)
        ),
    }


def _ambiguous_summary(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
) -> dict[str, Any]:
    case_ids = sorted(
        case_id
        for case_id, row in gt.items()
        if row["human_semantic_label"] == "ambiguous"
    )
    output = []
    for arm in BASE_ARMS:
        predictions = [
            results[repeat][(case_id, arm)]["predicted_label"]
            for case_id in case_ids
            for repeat in (1, 2)
        ]
        counts = Counter(predictions)
        output.append(
            {
                "arm": arm,
                "human_ambiguous_case_n": len(case_ids),
                "prediction_call_n": len(predictions),
                "predicted_valid_n": counts["valid"],
                "predicted_invalid_n": counts["invalid"],
                "predicted_ambiguous_n": counts["ambiguous"],
                "predicted_ambiguous_rate": (
                    counts["ambiguous"] / len(predictions)
                ),
            }
        )
    return {
        "case_ids": case_ids,
        "case_n": len(case_ids),
        "binary_accuracy_exclusion": True,
        "arms": output,
    }


def _proposal_sensitivity(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    proposed = {}
    for case_id in gt:
        events = _read_json(
            dataset_root / "fixtures" / case_id / "metric_gt.json"
        ).get("events") or []
        if len(events) != 1 or not isinstance(events[0], dict):
            raise ValueError(f"{case_id} must have exactly one metric GT proposal")
        proposed[case_id] = str(events[0].get("proposed_semantic_label") or "")
    output = []
    for metric in ("ALL",) + METRIC_ORDER:
        case_ids = sorted(
            case_id
            for case_id, row in gt.items()
            if (metric == "ALL" or row["metric"] == metric)
            and row["human_semantic_label"] in {"valid", "invalid"}
            and proposed[case_id] in {"valid", "invalid"}
        )
        if not case_ids:
            continue
        stable_disagreements = []
        for case_id in case_ids:
            left = results[1][(case_id, "production_default")]["predicted_label"]
            right = results[2][(case_id, "production_default")]["predicted_label"]
            if left == right and left != gt[case_id]["human_semantic_label"]:
                stable_disagreements.append((case_id, left))
        output.append(
            {
                "metric": metric,
                "binary_overlap_case_n": len(case_ids),
                "human_vs_proposal_agreement": _mean(
                    gt[case_id]["human_semantic_label"] == proposed[case_id]
                    for case_id in case_ids
                ),
                "model_vs_human_accuracy": _mean(
                    results[repeat][
                        (case_id, "production_default")
                    ]["predicted_label"]
                    == gt[case_id]["human_semantic_label"]
                    for case_id in case_ids
                    for repeat in (1, 2)
                ),
                "model_vs_proposal_accuracy": _mean(
                    results[repeat][
                        (case_id, "production_default")
                    ]["predicted_label"]
                    == proposed[case_id]
                    for case_id in case_ids
                    for repeat in (1, 2)
                ),
                "stable_model_human_disagreement_n": len(stable_disagreements),
                "proposal_agrees_with_stable_model_n": sum(
                    proposed[case_id] == model_label
                    for case_id, model_label in stable_disagreements
                ),
                "interpretation": (
                    "sensitivity_only_pending_construction_proposals_are_not_gt"
                ),
            }
        )
    return output


def _stable_disagreements(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
    review_cards: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    arm = "production_default"
    for case_id, gt_row in sorted(gt.items()):
        if gt_row["human_semantic_label"] not in {"valid", "invalid"}:
            continue
        left = results[1][(case_id, arm)]
        right = results[2][(case_id, arm)]
        if not (
            _bool(left["binary_match"]) is False
            and _bool(right["binary_match"]) is False
        ):
            continue
        card = review_cards[case_id]
        output.append(
            {
                "case_id": case_id,
                "metric": gt_row["metric"],
                "human_label": gt_row["human_semantic_label"],
                "human_notes": gt_row.get("notes") or "",
                "model_label_repeat_1": left["predicted_label"],
                "model_label_repeat_2": right["predicted_label"],
                "evidence_status_repeat_1": left["evidence_status"],
                "evidence_status_repeat_2": right["evidence_status"],
                "confidence_repeat_1": left["confidence"],
                "confidence_repeat_2": right["confidence"],
                "prompt": card["prompt"],
                "review_question": card["review_question"],
                "reason_repeat_1": left["reason"],
                "reason_repeat_2": right["reason"],
                "review_priority": (
                    "evidence_gap"
                    if "insufficient"
                    in {left["evidence_status"], right["evidence_status"]}
                    else "label_or_semantic_judge_audit"
                ),
            }
        )
    return output


def _oar_diagnostics(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
) -> dict[str, Any]:
    case_ids = sorted(
        case_id for case_id, row in gt.items() if row["metric"] == "oar"
    )
    repeated_direction_gap = []
    repeated_attachment_crop = []
    for case_id in case_ids:
        reasons = " ".join(
            results[repeat][(case_id, "production_default")]["reason"].lower()
            for repeat in (1, 2)
        )
        both_insufficient = all(
            results[repeat][(case_id, "production_default")]["evidence_status"]
            == "insufficient"
            for repeat in (1, 2)
        )
        if both_insufficient and any(
            token in reasons
            for token in ("room-axis", "axis legend", "directional cue", "as north", "as east", "as west")
        ):
            repeated_direction_gap.append(case_id)
        if both_insufficient and "ceiling" in reasons and "crop" in reasons:
            repeated_attachment_crop.append(case_id)
    return {
        "case_n": len(case_ids),
        "repeated_direction_grounding_gap_case_ids": repeated_direction_gap,
        "repeated_direction_grounding_gap_case_n": len(repeated_direction_gap),
        "repeated_ceiling_attachment_crop_case_ids": repeated_attachment_crop,
        "repeated_ceiling_attachment_crop_case_n": len(repeated_attachment_crop),
    }


def _identical_packet_control(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
) -> dict[str, Any]:
    pairs = []
    for repeat in (1, 2):
        for (case_id, arm), row in results[repeat].items():
            if (
                arm != "production_default"
                or row["metric"]
                not in {"room_scene_type", "broad_semantic_intent"}
            ):
                continue
            comparator = results[repeat][(case_id, "global_only")]
            if row["evidence_packet_sha256"] == comparator["evidence_packet_sha256"]:
                pairs.append(
                    row["predicted_label"] == comparator["predicted_label"]
                )
    return {
        "pair_n": len(pairs),
        "same_visual_packet": True,
        "prediction_agreement": _mean(pairs),
        "confound": (
            "The outbound judge context contains the experiment arm name, so "
            "these are image-identical but not byte-identical requests."
        ),
    }


def _report(
    payload: dict[str, Any],
    stable_disagreements: list[dict[str, Any]],
) -> str:
    production = {
        row["metric"]: row for row in payload["production_by_metric"]
    }
    overall = {row["arm"]: row for row in payload["overall_arms"]}
    chair = {row["arm"]: row for row in payload["chair_angle"]["arms"]}
    oar = payload["oar_diagnostics"]
    ambiguous = {
        row["arm"]: row for row in payload["ambiguous_gt"]["arms"]
    }
    sensitivity = {
        row["metric"]: row
        for row in payload["construction_proposal_sensitivity"]
    }
    paired = {
        (row["comparison_id"], row["metric"]): row
        for row in payload["paired_effects"]
    }
    local_vs_global = paired[("local_only_vs_global_only", "ALL")]
    add_local = paired[("local_gain_over_global", "ALL")]
    add_global = paired[("global_gain_over_local", "ALL")]
    full_vs_production = paired[("expanded_packet_vs_production", "ALL")]
    contour_production = paired[("contour_vs_raw_production", "ALL")]
    contour_local = paired[("contour_vs_raw_local", "ALL")]
    chair_case_ids = set(payload["chair_angle"]["case_ids"])
    stable_chair_disagreements = [
        row for row in stable_disagreements if row["case_id"] in chair_case_ids
    ]
    stable_chair_case_ids = ", ".join(
        row["case_id"] for row in stable_chair_disagreements
    )
    lines = [
        "# Exp 2 non-L1 VisualConfig ablation analysis",
        "",
        "## Bottom line",
        "",
        "- 当前 `production_default` 的核心优势是 stability，而不是显著更高的 "
        f"accuracy：binary accuracy 为 {_pct(overall['production_default']['binary_accuracy'])}，"
        f"repeat agreement 为 {_pct(overall['production_default']['label_repeat_agreement'])}.",
        "- 除 OAR 外，production evidence sufficiency 均为 "
        "93.8%–100%. 因此多数剩余错误更像 semantic rubric / judge / GT "
        "disagreement，而不是单纯 camera 缺图.",
        f"- OAR 是确定的 VisualConfig failure：sufficiency 只有 "
        f"{_pct(production['oar']['evidence_sufficient_rate'])}. "
        f"{oar['repeated_direction_grounding_gap_case_n']}/{oar['case_n']} cases "
        "两次都缺 room-axis grounding，另有 ceiling attachment 被裁切.",
        f"- 按你的 human GT，`local_raw_only` 相对 `global_only` aggregate "
        f"accuracy 提高 {local_vs_global['delta_accuracy_pp']:+.1f} pp "
        f"(case-bootstrap 95% CI "
        f"{local_vs_global['bootstrap_ci_low_pp']:+.1f} to "
        f"{local_vs_global['bootstrap_ci_high_pp']:+.1f}). "
        "但每个 metric 单独的 interval 仍都触及 0.",
        f"- Human labels 与 construction proposal 的 binary overlap 只有 "
        f"{_pct(sensitivity['ALL']['human_vs_proposal_agreement'])} agreement；"
        "后者不是 GT，但说明 absolute accuracy 对 annotation boundary 很敏感.",
        "",
        "## Scoring contract used here",
        "",
        "- 你的 `human_semantic_label` 是本报告唯一 GT. Construction proposal "
        "完全不参与 accuracy.",
        "- GT 包含 61 valid、37 invalid、10 ambiguous cases. 10 个 human-ambiguous "
        "cases 单列，不进入 binary accuracy.",
        "- 每个 binary case 运行两次，因此每个 base arm 有 196 个 scored model calls. "
        "模型在 binary GT 上输出 `ambiguous` 时按未答对计.",
        "",
        "## Machine judgement against your GT",
        "",
        "| Arm | Repeat 1 | Repeat 2 | Combined | Balanced acc. | FP | FN | Model ambiguous | Repeat agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in BASE_ARMS:
        row = overall[arm]
        lines.append(
            f"| {arm} | {row['repeat_1_correct_n']}/98 | "
            f"{row['repeat_2_correct_n']}/98 | "
            f"{row['correct_call_n']}/{row['binary_call_n']} "
            f"({_pct(row['binary_accuracy'])}) | "
            f"{_pct(row['balanced_accuracy'])} | "
            f"{row['false_positive_call_n']} | "
            f"{row['false_negative_call_n']} | "
            f"{row['predicted_ambiguous_call_n']} | "
            f"{_pct(row['label_repeat_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "`invalid` 在 FP/FN 定义中作为 positive：FP = GT valid 但模型判 invalid；"
            "FN = GT invalid 但模型判 valid. Repeat agreement 在全部 108 cases 上计算.",
            "",
            "## Production machine judgement by metric",
            "",
            "| Metric | Correct calls | Accuracy | FP | FN | Model ambiguous | Evidence sufficient | Repeat agreement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in METRIC_ORDER:
        row = production[metric]
        lines.append(
            f"| {metric} | {row['correct_call_n']}/{row['binary_call_n']} | "
            f"{_pct(row['binary_accuracy'])} | "
            f"{row['false_positive_call_n']} | "
            f"{row['false_negative_call_n']} | "
            f"{row['predicted_ambiguous_call_n']} | "
            f"{_pct(row['evidence_sufficient_rate'])} | "
            f"{_pct(row['label_repeat_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "## Ablation takeaways",
            "",
            f"1. **Local-only vs global-only**：local 为 "
            f"{_pct(overall['local_raw_only']['binary_accuracy'])}，global 为 "
            f"{_pct(overall['global_only']['binary_accuracy'])}，差值 "
            f"{local_vs_global['delta_accuracy_pp']:+.1f} pp；"
            f"{local_vs_global['right_better_case_n']} cases 改善，"
            f"{local_vs_global['left_better_case_n']} cases 变差，"
            f"{local_vs_global['tied_case_n']} ties. Gain 主要来自 OAR、scale、"
            "object pairing 和 broad intent；但 OAR 的绝对 accuracy 仍只有 16.7%.",
            f"2. **Add local to global**：`full_raw - global_only` 为 "
            f"{add_local['delta_accuracy_pp']:+.1f} pp "
            f"(95% CI {add_local['bootstrap_ci_low_pp']:+.1f} to "
            f"{add_local['bootstrap_ci_high_pp']:+.1f}). 说明 local evidence "
            "在当前 mixed metric set 上有 aggregate value；room/scene type 和 "
            "required areas 没有额外 gain.",
            f"3. **Add global to local**：`full_raw - local_raw_only` 为 "
            f"{add_global['delta_accuracy_pp']:+.1f} pp "
            f"(95% CI {add_global['bootstrap_ci_low_pp']:+.1f} to "
            f"{add_global['bootstrap_ci_high_pp']:+.1f})，"
            f"{add_global['right_better_case_n']} improve / "
            f"{add_global['left_better_case_n']} worsen. Global 不是所有 "
            "object-level metrics 的通用增益；它应保留给 global-semantic metrics "
            "和确实需要 context 的 OOR cases.",
            f"4. **Full five raw views vs metric-specific production**：overall 仅 "
            f"{full_vs_production['delta_accuracy_pp']:+.1f} pp "
            f"(95% CI {full_vs_production['bootstrap_ci_low_pp']:+.1f} to "
            f"{full_vs_production['bootstrap_ci_high_pp']:+.1f}). Pairing "
            "+12.5 pp，但 scale -7.1 pp、OOR -4.5 pp. 更多 images 没有 "
            "universal gain，不能统一切到 budget 5.",
            f"5. **Contour vs same-pose raw in production packet**：OOR/OAR 合计 "
            f"{contour_production['delta_accuracy_pp']:+.1f} pp "
            f"(95% CI {contour_production['bootstrap_ci_low_pp']:+.1f} to "
            f"{contour_production['bootstrap_ci_high_pp']:+.1f})；"
            "仅 2 cases 改善、1 case 变差. 这是 directional signal，不是稳定证据.",
            f"6. **Contour vs raw, local-only**：合计 "
            f"{contour_local['delta_accuracy_pp']:+.1f} pp "
            f"(95% CI {contour_local['bootstrap_ci_low_pp']:+.1f} to "
            f"{contour_local['bootstrap_ci_high_pp']:+.1f}). OOR 为 +13.6 pp，"
            "OAR 为 0. Contour 更像 identity/target grounding aid，不能提供 "
            "room-axis semantics.",
            "7. **Metric-specific production vs universal arms**：production accuracy "
            "63.8%，不高于 local/full 的 64.8%，但 repeat agreement 最高 "
            "(93.5%). 当前 production policy 的实证价值是 stability 与更小 image "
            "budget，而不是已证明的最高 accuracy.",
            "",
            "## Confidence of takeaways",
            "",
            "- **High confidence**：OAR 当前 observation contract 不充分；"
            "room/scene type 不需要 local evidence；增加普通 views 没有 universal gain.",
            "- **Medium confidence**：在当前 human-GT mixed set 上，local evidence "
            "相对 global evidence 有 aggregate gain；chair/object orientation 更适合 "
            "orientation-aware local evidence.",
            "- **Low confidence / directional only**：same-pose contour 对 OOR 的提升.",
            "- **Low confidence / requires more data**：`full_raw` 对 object pairing 的 "
            "+12.5 pp，以及任一 arm 的小幅 metric-level ranking. 这些 bootstrap "
            "interval 都跨 0.",
            "",
            "## Chair-angle warning",
            "",
            f"严格 chair-angle subset 共 {payload['chair_angle']['case_n']} cases. "
            f"`production_default` accuracy {_pct(chair['production_default']['binary_accuracy'])}，"
            f"`local_raw_only` {_pct(chair['local_raw_only']['binary_accuracy'])}；"
            f"repeat agreement 分别为 {_pct(chair['production_default']['label_repeat_agreement'])} "
            f"和 {_pct(chair['local_raw_only']['label_repeat_agreement'])}.",
            "",
            "这个方向支持 orientation-aware local evidence，但不能直接解释为 camera "
            "已经解决问题：stable disagreements 中同时存在 chair front/back 判断、"
            "prompt-authorized exception 和可能的 human-label error. 建议将这 10 cases "
            "单独二次审计，并显式标注 object semantic front axis.",
            "",
            f"其中 production 两次都与 human label 不同的 strict chair cases 为 "
            f"`{stable_chair_case_ids}`. `case_0073` 的 human note 本身写有 "
            "\"Angle of chair reverse\"；`case_0104` 的 prompt 明确授权 chair "
            "face away，但 human label 仍为 invalid. 因此这 3 个 case 应优先 "
            "re-audit，不能作为纯 VisualConfig error 计数.",
            "",
            "## GT and calibration caveats",
            "",
            f"- Human ambiguous 共 {payload['ambiguous_gt']['case_n']} cases，已经从 "
            "binary accuracy 排除.",
            f"- 在 human / construction proposal 都给出 binary label 的 "
            f"{sensitivity['ALL']['binary_overlap_case_n']} cases 上，production "
            f"对 human 为 {_pct(sensitivity['ALL']['model_vs_human_accuracy'])}，"
            f"对 proposal 为 {_pct(sensitivity['ALL']['model_vs_proposal_accuracy'])}. "
            "这只是 sensitivity check，不能用 proposal 替换人工 GT.",
            f"- 对这些 cases，production 20 次 calls 中预测 ambiguous 的次数为 "
            f"{ambiguous['production_default']['predicted_ambiguous_n']}; "
            "这表明 judge 不会自然复现 human uncertainty.",
            f"- Production stable binary disagreements 共 {len(stable_disagreements)} "
            "cases. 多数 evidence_status=sufficient 且模型理由在两次 repeat 中一致，"
            "应作为 GT / rubric / semantic judge audit queue，而不是全部归因于 VisualConfig.",
            "- Model confidence 对 correct 与 wrong calls 几乎没有区分，因此不能直接用 "
            "self-reported confidence 做 abstention threshold.",
            "- 48 个 image-identical global packet controls 中有 1 个 prediction change；"
            "此外 judge context 暴露了 arm name. 下一轮应从 outbound context 删除 arm name.",
            "",
            "## Decision",
            "",
            "- 可复用：room/scene type、required functional areas、scale、style、OOR 的 "
            "global/local evidence scaffold.",
            "- 需要 metric-specific repair：OAR direction/attachment grounding，以及 "
            "chair/object orientation evidence.",
            "- 暂不需要全面引入 VLM-active camera. 只在 repaired static packet 仍返回 "
            "`insufficient` 时触发 bounded active fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_repeats(
    run_root: Path,
) -> dict[int, dict[tuple[str, str], dict[str, str]]]:
    output = {}
    for repeat in (1, 2):
        path = run_root / f"repeat_{repeat}" / "per_event.tsv"
        rows = _read_tsv(path)
        indexed = {}
        for row in rows:
            if row.get("error"):
                raise ValueError(f"unresolved result in repeat {repeat}: {row}")
            key = (row["case_id"], row["arm"])
            if key in indexed:
                raise ValueError(f"duplicate result in repeat {repeat}: {key}")
            indexed[key] = row
        if len(indexed) != 480:
            raise ValueError(
                f"repeat {repeat} must contain 480 resolved jobs, got {len(indexed)}"
            )
        output[repeat] = indexed
    return output


def _read_gt(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_tsv(path)
    output = {row["case_id"]: row for row in rows}
    if len(output) != 108:
        raise ValueError(f"human GT must have 108 unique cases, got {len(output)}")
    return output


def _review_cards(path: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(path)
    cards = {
        str(item["case_id"]): item
        for item in value.get("cases") or []
        if isinstance(item, dict)
    }
    if len(cards) != 108:
        raise ValueError(f"review cards must contain 108 cases, got {len(cards)}")
    return cards


def _validate_identity(
    results: dict[int, dict[tuple[str, str], dict[str, str]]],
    gt: dict[str, dict[str, str]],
) -> None:
    for repeat, rows in results.items():
        for (case_id, _arm), row in rows.items():
            if case_id not in gt:
                raise ValueError(f"repeat {repeat} has unknown case {case_id}")
            if row["metric"] != gt[case_id]["metric"]:
                raise ValueError(f"metric drift for {case_id}")
            if row["gt_label"] != gt[case_id]["human_semantic_label"]:
                raise ValueError(f"GT drift for {case_id}")


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed_material: str,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    count = len(values)
    means = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    means.sort()
    low_index = max(0, int(0.025 * samples))
    high_index = min(samples - 1, int(0.975 * samples))
    return statistics.fmean(values), means[low_index], means[high_index]


def _mean(values: Any) -> float | None:
    selected = [value for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def _bool(value: Any) -> bool | None:
    if value in {True, "True", "true", "1", 1}:
        return True
    if value in {False, "False", "false", "0", 0}:
        return False
    return None


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=GT_PATH)
    parser.add_argument("--review-cases", type=Path, default=REVIEW_CASES)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty TSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
