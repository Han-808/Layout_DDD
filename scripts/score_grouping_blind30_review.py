#!/usr/bin/env python3
"""Score a completed blind grouping review after unblinding the method key."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.grouping_blind30_contracts import (
    BLIND_LABELS,
    atomic_write_json,
    load_experiment_config,
    read_json,
)
from scripts.serve_grouping_blind30_review import (
    HUMAN_REVIEW_SCHEMA_VERSION,
    validate_review_payload,
)


SCORE_SCHEMA_VERSION = "grouping_blind30_scores_v1"
BACKENDS = ("topology", "anchor", "vlm")
QUALITY_SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}
OBJECT_COUNT_GROUPS = ("<11 objects", "11–30 objects", ">30 objects")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiments"
        / "grouping_blind30_gpt56_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write a provisional score using available labels instead of "
            "requiring all scenes and variants to be reviewed."
        ),
    )
    args = parser.parse_args()
    config, paths = load_experiment_config(
        args.config,
        repo_root=PROJECT_ROOT,
        output_override=args.output_root,
    )
    review_data = read_json(paths.review_root / "review_data.json")
    review_path = paths.output_root / "human_reviews" / "blind_reviews.json"
    review_payload = read_json(review_path)
    method_key = read_json(paths.method_key)
    scores = calculate_grouping_scores(
        review_data=review_data,
        review_payload=review_payload,
        method_key=method_key,
        allow_incomplete=args.allow_incomplete,
    )
    output_path = (
        paths.output_root / "human_reviews" / "grouping_scores.json"
    )
    atomic_write_json(output_path, scores)
    print(
        json.dumps(
            {
                "score_path": str(output_path),
                "complete": scores["complete"],
                "overall": scores["overall"],
                "by_backend": scores["by_backend"],
            },
            indent=2,
        )
    )


def calculate_grouping_scores(
    *,
    review_data: dict[str, Any],
    review_payload: dict[str, Any],
    method_key: dict[str, Any],
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Return post-review scores, keeping blind labels out of the UI path."""

    experiment_id = _required_text(
        review_data.get("experiment_id"),
        "review_data.experiment_id",
    )
    if review_payload.get("experiment_id") not in (None, experiment_id):
        raise ValueError("review payload experiment_id does not match")
    normalized_payload = _normalize_review_payload(
        review_payload,
        experiment_id=experiment_id,
        review_data=review_data,
    )
    cases = review_data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("review_data.cases must be a non-empty list")
    case_ids = tuple(
        _required_text(item.get("case_id"), "case_id") for item in cases
    )
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("review_data.cases contains duplicate case IDs")

    mapping = _validate_method_key(
        method_key,
        experiment_id=experiment_id,
        expected_case_ids=case_ids,
    )
    answers = normalized_payload["answers"]
    missing_cases = [case_id for case_id in case_ids if case_id not in answers]
    missing_labels: list[dict[str, str]] = []
    unreviewed_cases: list[str] = []
    for case_id in case_ids:
        answer = answers.get(case_id)
        if not isinstance(answer, dict):
            missing_labels.extend(
                {"case_id": case_id, "blind_label": label}
                for label in BLIND_LABELS
            )
            continue
        if answer.get("reviewed") is not True:
            unreviewed_cases.append(case_id)
        variants = answer.get("variants")
        if not isinstance(variants, dict):
            missing_labels.extend(
                {"case_id": case_id, "blind_label": label}
                for label in BLIND_LABELS
            )
            continue
        for label in BLIND_LABELS:
            quality = variants.get(label, {}).get("quality")
            if quality == "":
                missing_labels.append(
                    {"case_id": case_id, "blind_label": label}
                )

    complete = not missing_cases and not missing_labels and not unreviewed_cases
    if not allow_incomplete and not complete:
        problems = []
        if missing_cases:
            problems.append(f"missing cases={len(missing_cases)}")
        if missing_labels:
            problems.append(f"missing quality labels={len(missing_labels)}")
        if unreviewed_cases:
            problems.append(f"unreviewed cases={len(unreviewed_cases)}")
        raise ValueError(
            "blind review is not complete; " + ", ".join(problems)
        )

    accumulators = {
        backend: _new_accumulator(expected=len(case_ids))
        for backend in BACKENDS
    }
    group_by_case = {
        _required_text(item.get("case_id"), "case_id"): _object_count_group(
            item.get("object_count")
        )
        for item in cases
    }
    group_case_counts: dict[str, int] = defaultdict(int)
    for group in group_by_case.values():
        group_case_counts[group] += 1
    by_object_count_group = {
        group: {
            backend: _new_accumulator(expected=group_case_counts[group])
            for backend in BACKENDS
        }
        for group in OBJECT_COUNT_GROUPS
    }
    for case_id in case_ids:
        answer = answers.get(case_id)
        if not isinstance(answer, dict):
            continue
        variants = answer.get("variants")
        if not isinstance(variants, dict):
            continue
        for blind_label in BLIND_LABELS:
            method = mapping[case_id][blind_label]
            quality = variants[blind_label].get("quality")
            accumulator = accumulators[method]
            grouped_accumulator = by_object_count_group[
                group_by_case[case_id]
            ][method]
            accumulator["labels_available"] += int(quality != "")
            grouped_accumulator["labels_available"] += int(quality != "")
            if quality == "unclear":
                accumulator["unscored_count"] += 1
                grouped_accumulator["unscored_count"] += 1
                continue
            if quality not in QUALITY_SCORES:
                if quality == "":
                    accumulator["missing_count"] += 1
                    grouped_accumulator["missing_count"] += 1
                    continue
                raise ValueError(
                    f"unsupported quality {quality!r} for "
                    f"{case_id}/{blind_label}"
                )
            _add_score(accumulator, quality)
            _add_score(grouped_accumulator, quality)

    overall = _new_accumulator(expected=len(case_ids) * len(BLIND_LABELS))
    for accumulator in accumulators.values():
        _merge_accumulator(overall, accumulator)
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_fingerprint": method_key.get("dataset_fingerprint"),
        "complete": complete,
        "allow_incomplete": allow_incomplete,
        "scoring_rule": {
            "correct": 1.0,
            "partially_correct": 0.5,
            "incorrect": 0.0,
            "unclear": None,
            "blank": None,
            "normalization": "score_sum / scored_label_count",
        },
        "review_coverage": {
            "expected_cases": len(case_ids),
            "answered_cases": len(answers),
            "reviewed_cases": sum(
                bool(answers.get(case_id, {}).get("reviewed"))
                for case_id in case_ids
                if isinstance(answers.get(case_id), dict)
            ),
            "missing_cases": missing_cases,
            "missing_quality_labels": missing_labels,
            "unreviewed_cases": unreviewed_cases,
        },
        "overall": _finalize_accumulator(overall),
        "by_backend": {
            backend: _finalize_accumulator(accumulators[backend])
            for backend in BACKENDS
        },
        "by_object_count_group": {
            group: {
                "scene_count": group_case_counts[group],
                "by_backend": {
                    backend: _finalize_accumulator(group_accumulator)
                    for backend, group_accumulator in values.items()
                },
            }
            for group, values in by_object_count_group.items()
        },
    }


def _normalize_review_payload(
    value: dict[str, Any],
    *,
    experiment_id: str,
    review_data: dict[str, Any],
) -> dict[str, Any]:
    # Early browser/localStorage exports from the first UI version contained
    # only {"answers": ...}. Keep that artifact readable without weakening
    # the current server validator for new writes.
    candidate = dict(value)
    if "schema_version" not in candidate:
        candidate = {
            "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
            "experiment_id": candidate.get("experiment_id", experiment_id),
            "answers": candidate.get("answers", candidate),
        }
    return validate_review_payload(candidate, review_data=review_data)


def _validate_method_key(
    value: dict[str, Any],
    *,
    experiment_id: str,
    expected_case_ids: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    if value.get("experiment_id") != experiment_id:
        raise ValueError("private method key experiment_id does not match")
    cases = value.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("private method key cases must be an object")
    if set(cases) != set(expected_case_ids):
        raise ValueError("private method key case set does not match review")
    result: dict[str, dict[str, str]] = {}
    for case_id in expected_case_ids:
        entry = cases[case_id]
        if not isinstance(entry, dict) or set(entry) != set(BLIND_LABELS):
            raise ValueError(
                f"private method key {case_id} must map A, B, and C"
            )
        if set(entry.values()) != set(BACKENDS):
            raise ValueError(
                f"private method key {case_id} must contain all backends"
            )
        result[case_id] = {label: str(entry[label]) for label in BLIND_LABELS}
    return result


def _new_accumulator(*, expected: int) -> dict[str, Any]:
    return {
        "expected_labels": expected,
        "labels_available": 0,
        "scored_label_count": 0,
        "correct_count": 0,
        "partially_correct_count": 0,
        "incorrect_count": 0,
        "unscored_count": 0,
        "missing_count": 0,
        "score_sum": 0.0,
    }


def _add_score(accumulator: dict[str, Any], quality: str) -> None:
    accumulator["scored_label_count"] += 1
    accumulator[f"{quality}_count"] += 1
    accumulator["score_sum"] += QUALITY_SCORES[quality]


def _merge_accumulator(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key in (
        "labels_available",
        "scored_label_count",
        "correct_count",
        "partially_correct_count",
        "incorrect_count",
        "unscored_count",
        "missing_count",
    ):
        target[key] += source[key]
    target["score_sum"] += source["score_sum"]


def _finalize_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    scored = accumulator["scored_label_count"]
    expected = accumulator["expected_labels"]
    return {
        **accumulator,
        "normalized_score": (
            accumulator["score_sum"] / scored if scored else None
        ),
        "coverage": scored / expected if expected else None,
    }


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _object_count_group(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("review_data.object_count must be a non-negative integer")
    if value < 11:
        return "<11 objects"
    if value < 31:
        return "11–30 objects"
    return ">30 objects"


if __name__ == "__main__":
    main()
