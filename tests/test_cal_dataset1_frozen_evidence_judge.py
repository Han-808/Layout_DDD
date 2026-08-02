from __future__ import annotations

import json

from scripts import judge_cal_dataset1_camera_evidence as module


def _result(*, arm: str, predicted: str | None, metric: str = "collision") -> dict:
    return {
        "case_id": "case_001",
        "metric": metric,
        "event_id": "obj_a|obj_b",
        "severity_class": "obvious",
        "arm": arm,
        "gt_label": "invalid",
        "predicted_label": predicted,
        "resolved": predicted in {"valid", "invalid"},
        "match": predicted == "invalid",
        "confidence": 0.8 if predicted else None,
        "elapsed_seconds": 1.0 if predicted else None,
        "error": None if predicted else "EndpointHTTPError: failed",
    }


def _ambiguous_result(*, arm: str, predicted: str | None) -> dict:
    result = _result(arm=arm, predicted=predicted)
    result.update(
        {
            "severity_class": "edge",
            "gt_label": "ambiguous",
            "match": None,
        }
    )
    return result


def test_summary_reports_invalid_recall_without_claiming_fp_rate() -> None:
    results = [
        _result(arm="fixed_global_highlight", predicted="valid"),
        _result(arm="metric_local_highlight", predicted="invalid"),
    ]

    rows = module._summary_rows(results)
    fixed = next(
        row
        for row in rows
        if row["arm"] == "fixed_global_highlight" and row["group"] == "overall"
    )
    local = next(
        row
        for row in rows
        if row["arm"] == "metric_local_highlight" and row["group"] == "overall"
    )

    assert fixed["invalid_recall_all"] == 0.0
    assert fixed["false_negatives"] == 1
    assert local["invalid_recall_all"] == 1.0
    assert "false_positives" not in fixed


def test_ambiguous_summary_reports_tendency_without_binary_accuracy() -> None:
    results = [
        _ambiguous_result(
            arm="fixed_global_highlight",
            predicted="invalid",
        ),
        _ambiguous_result(
            arm="metric_local_highlight",
            predicted="valid",
        ),
    ]

    rows = module._summary_rows(results)
    fixed = next(
        row
        for row in rows
        if row["arm"] == "fixed_global_highlight" and row["group"] == "overall"
    )
    local = next(
        row
        for row in rows
        if row["arm"] == "metric_local_highlight" and row["group"] == "overall"
    )

    assert fixed["scored_total"] == 0
    assert fixed["correct"] == 0
    assert fixed["accuracy_scored"] is None
    assert fixed["invalid_gt_total"] == 0
    assert fixed["invalid_recall_all"] is None
    assert fixed["ambiguous_total"] == 1
    assert fixed["ambiguous_invalid_rate"] == 1.0
    assert local["ambiguous_invalid_rate"] == 0.0


def test_paired_summary_attributes_one_recovery_to_local_arm() -> None:
    results = [
        _result(arm="fixed_global_highlight", predicted="valid"),
        _result(arm="metric_local_highlight", predicted="invalid"),
    ]

    row = next(
        item
        for item in module._paired_rows(results)
        if item["group_type"] == "overall"
    )

    assert row["fixed_only_correct"] == 0
    assert row["local_only_correct"] == 1
    assert row["local_minus_fixed_correct"] == 1


def test_paired_summary_treats_ambiguous_events_as_verdict_transitions() -> None:
    results = [
        _ambiguous_result(
            arm="fixed_global_highlight",
            predicted="invalid",
        ),
        _ambiguous_result(
            arm="metric_local_highlight",
            predicted="valid",
        ),
    ]

    row = next(
        item
        for item in module._paired_rows(results)
        if item["group_type"] == "overall"
    )

    assert row["scored_pairs"] == 0
    assert row["ambiguous_pairs"] == 1
    assert row["ambiguous_resolved_pairs"] == 1
    assert row["both_incorrect"] == 0
    assert row["ambiguous_fixed_invalid_local_valid"] == 1
    assert row["ambiguous_verdict_agreement"] == 0.0


def test_resume_requires_exact_contract_and_successful_binary_verdict(tmp_path) -> None:
    contract = {"schema_version": "contract", "arm": "metric_local_highlight"}
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": module.EVENT_RESULT_SCHEMA_VERSION,
                "contract_sha256": module._json_sha256(contract),
                "predicted_label": "invalid",
                "error": None,
                "judgement": {"verdict": "invalid"},
            }
        )
    )

    assert module._result_ready(result_path, contract) is True
    assert module._result_ready(result_path, {**contract, "arm": "changed"}) is False
