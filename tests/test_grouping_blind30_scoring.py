from __future__ import annotations

import pytest

from scripts.score_grouping_blind30_review import (
    _object_count_group,
    calculate_grouping_scores,
)


def _answer(
    qualities: dict[str, str],
    *,
    reviewed: bool = True,
) -> dict:
    return {
        "reviewed": reviewed,
        "best_result": "",
        "notes": "",
        "variants": {
            label: {"quality": qualities[label], "notes": ""}
            for label in ("A", "B", "C")
        },
    }


def _fixtures() -> tuple[dict, dict, dict]:
    review_data = {
        "experiment_id": "grouping_test",
        "cases": [
            {"case_id": "case_001", "object_count": 6},
            {"case_id": "case_002", "object_count": 32},
        ],
    }
    review_payload = {
        "schema_version": "grouping_blind30_human_reviews_v1",
        "experiment_id": "grouping_test",
        "answers": {
            "case_001": _answer(
                {
                    "A": "correct",
                    "B": "partially_correct",
                    "C": "incorrect",
                }
            ),
            "case_002": _answer(
                {
                    "A": "partially_correct",
                    "B": "incorrect",
                    "C": "correct",
                }
            ),
        },
    }
    method_key = {
        "experiment_id": "grouping_test",
        "dataset_fingerprint": "dataset-test",
        "cases": {
            "case_001": {"A": "topology", "B": "anchor", "C": "vlm"},
            "case_002": {"A": "vlm", "B": "topology", "C": "anchor"},
        },
    }
    return review_data, review_payload, method_key


def test_scores_quality_labels_after_unblinding() -> None:
    review_data, review_payload, method_key = _fixtures()

    result = calculate_grouping_scores(
        review_data=review_data,
        review_payload=review_payload,
        method_key=method_key,
    )

    assert result["complete"] is True
    assert result["scoring_rule"]["correct"] == 1.0
    assert result["scoring_rule"]["partially_correct"] == 0.5
    assert result["scoring_rule"]["incorrect"] == 0.0
    assert result["by_backend"]["topology"]["score_sum"] == 1.0
    assert result["by_backend"]["topology"]["normalized_score"] == 0.5
    assert result["by_backend"]["anchor"]["score_sum"] == 1.5
    assert result["by_backend"]["anchor"]["normalized_score"] == 0.75
    assert result["by_backend"]["vlm"]["score_sum"] == 0.5
    assert result["by_backend"]["vlm"]["normalized_score"] == 0.25
    assert result["overall"]["scored_label_count"] == 6
    assert result["overall"]["normalized_score"] == pytest.approx(
        3.0 / 6.0
    )
    assert result["by_object_count_group"]["<11 objects"][
        "by_backend"
    ]["topology"]["coverage"] == 1.0


def test_strict_scoring_rejects_unfinished_review() -> None:
    review_data, review_payload, method_key = _fixtures()
    review_payload["answers"]["case_002"]["variants"]["C"][
        "quality"
    ] = ""
    review_payload["answers"]["case_002"]["reviewed"] = False

    with pytest.raises(ValueError, match="blind review is not complete"):
        calculate_grouping_scores(
            review_data=review_data,
            review_payload=review_payload,
            method_key=method_key,
        )

    provisional = calculate_grouping_scores(
        review_data=review_data,
        review_payload=review_payload,
        method_key=method_key,
        allow_incomplete=True,
    )
    assert provisional["complete"] is False
    assert provisional["review_coverage"]["missing_quality_labels"] == [
        {"case_id": "case_002", "blind_label": "C"}
    ]
    assert provisional["by_backend"]["anchor"]["missing_count"] == 1


def test_unclear_is_reported_but_not_treated_as_zero() -> None:
    review_data, review_payload, method_key = _fixtures()
    review_payload["answers"]["case_002"]["variants"]["C"][
        "quality"
    ] = "unclear"

    result = calculate_grouping_scores(
        review_data=review_data,
        review_payload=review_payload,
        method_key=method_key,
    )
    anchor = result["by_backend"]["anchor"]
    assert anchor["unscored_count"] == 1
    assert anchor["scored_label_count"] == 1
    assert anchor["score_sum"] == 0.5
    assert anchor["normalized_score"] == 0.5


def test_legacy_answers_only_payload_is_supported() -> None:
    review_data, review_payload, method_key = _fixtures()
    legacy_payload = {"answers": review_payload["answers"]}

    result = calculate_grouping_scores(
        review_data=review_data,
        review_payload=legacy_payload,
        method_key=method_key,
    )

    assert result["complete"] is True


def test_object_count_group_boundaries_match_review_protocol() -> None:
    assert _object_count_group(10) == "<11 objects"
    assert _object_count_group(11) == "11–30 objects"
    assert _object_count_group(30) == "11–30 objects"
    assert _object_count_group(31) == ">30 objects"
