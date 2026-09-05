from __future__ import annotations

from scripts.judge_cal_dataset1_support_topk import (
    _paired_rows as support_topk_paired_rows,
    _summary_rows as support_topk_summary_rows,
)
from scripts.prepare_cal_dataset1_support_topk import _topk_selection
from scripts.summarize_cal_dataset1_valid_fp import _summary_rows as valid_fp_summary_rows


def test_valid_fp_summary_keeps_unresolved_in_all_event_denominator() -> None:
    rows = [
        {
            "metric": "support",
            "predicted_label": "valid",
            "resolved": 1,
            "false_positive": 0,
            "route": "direct_valid_contact",
            "vlm_adjudicated": 0,
        },
        {
            "metric": "support",
            "predicted_label": "invalid",
            "resolved": 1,
            "false_positive": 1,
            "route": "vlm_adjudicated",
            "vlm_adjudicated": 1,
        },
        {
            "metric": "support",
            "predicted_label": "unresolved",
            "resolved": 0,
            "false_positive": 0,
            "route": None,
            "vlm_adjudicated": 0,
        },
    ]

    overall = valid_fp_summary_rows(rows)[0]

    assert overall["valid_gt_total"] == 3
    assert overall["resolved"] == 2
    assert overall["unresolved"] == 1
    assert overall["false_positives"] == 1
    assert overall["fp_rate_all"] == 1 / 3
    assert overall["fp_rate_resolved"] == 1 / 2
    assert overall["vlm_fp_rate_all"] == 1.0


def test_support_topk_reconstruction_is_nested_with_frozen_top2() -> None:
    candidates = [
        {"id": "a", "azimuth_degrees": 0.0, "elevation_degrees": 5.0},
        {"id": "b", "azimuth_degrees": 180.0, "elevation_degrees": 5.0},
        {"id": "c", "azimuth_degrees": 90.0, "elevation_degrees": 5.0},
        {"id": "d", "azimuth_degrees": 270.0, "elevation_degrees": 5.0},
    ]
    ranked = [
        {"id": "a", "base_score": 1.0, "usable": False},
        {"id": "b", "base_score": 0.9, "usable": False},
        {"id": "c", "base_score": 0.8, "usable": False},
        {"id": "d", "base_score": 0.7, "usable": False},
    ]
    manifest = {
        "selection": {
            "selected_view_ids": ["a", "b"],
            "ranking": {
                "selector": "support_contact_plane_visibility_rank_v1",
                "ranked": ranked,
            },
        }
    }

    selected, reconstruction = _topk_selection(candidates, manifest, {})

    assert [item["id"] for item in selected] == ["a", "b", "c"]
    assert reconstruction["selected_view_ids"] == ["a", "b", "c"]
    assert reconstruction["reconstructed_from_frozen_ranking"] is True


def test_support_topk_summary_separates_invalid_accuracy_from_ambiguous_tendency() -> None:
    results = []
    for arm, invalid_prediction, ambiguous_prediction in [
        ("support_top1", "valid", "invalid"),
        ("support_top2", "invalid", "invalid"),
        ("support_top3", "invalid", "valid"),
    ]:
        results.extend([
            {
                "case_id": "invalid_case",
                "event_id": "obj_0",
                "severity_class": "subtle",
                "arm": arm,
                "gt_label": "invalid",
                "predicted_label": invalid_prediction,
                "resolved": True,
                "confidence": 0.8,
                "elapsed_seconds": 1.0,
                "error": None,
            },
            {
                "case_id": "edge_case",
                "event_id": "obj_1",
                "severity_class": "edge",
                "arm": arm,
                "gt_label": "ambiguous",
                "predicted_label": ambiguous_prediction,
                "resolved": True,
                "confidence": 0.7,
                "elapsed_seconds": 1.0,
                "error": None,
            },
        ])

    summary = support_topk_summary_rows(results)
    overall = {
        row["arm"]: row
        for row in summary
        if row["group_type"] == "overall"
    }
    paired = support_topk_paired_rows(
        results,
        ("support_top1", "support_top2", "support_top3"),
    )

    assert overall["support_top1"]["invalid_recall_all"] == 0.0
    assert overall["support_top2"]["invalid_recall_all"] == 1.0
    assert overall["support_top3"]["ambiguous_invalid_rate"] == 0.0
    top1_top2_invalid = next(
        row
        for row in paired
        if row["lower_arm"] == "support_top1"
        and row["higher_arm"] == "support_top2"
        and row["group"] == "invalid"
    )
    assert top1_top2_invalid["lower_valid_higher_invalid"] == 1
