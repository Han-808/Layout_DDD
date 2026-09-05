from __future__ import annotations

import pytest

from benchmark.evaluator.scoring import (
    project_metric_events,
    score_l3_metric_report,
)


def _ids(count: int) -> tuple[str, ...]:
    return tuple(f"obj_{index}" for index in range(count))


@pytest.mark.parametrize("verdict", [None, "ambiguous"])
def test_l3_scoring_rejects_missing_or_unknown_verdict(
    verdict: str | None,
) -> None:
    judgement = {"defects": []}
    if verdict is not None:
        judgement["verdict"] = verdict

    with pytest.raises(ValueError, match="verdict"):
        score_l3_metric_report(
            "style_consistency",
            {"judgement": judgement},
            ordered_object_ids=_ids(2),
        )


def test_l3_scoring_rejects_valid_verdict_with_defects() -> None:
    with pytest.raises(ValueError, match="valid.*defects|defects.*valid"):
        score_l3_metric_report(
            "style_consistency",
            {
                "judgement": {
                    "verdict": "valid",
                    "defects": [
                        {
                            "category": "style_outlier",
                            "severity": "gross",
                            "target_ids": ["obj_0"],
                            "scope": "visible_design_language",
                            "relation": "style conflict",
                            "reason": "A material style conflict is visible.",
                        }
                    ],
                }
            },
            ordered_object_ids=_ids(2),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("category", "unknown_category", "category"),
        ("severity", "unknown_severity", "severity"),
    ],
)
def test_l3_scoring_rejects_explicit_unknown_categorical_values(
    field: str,
    value: str,
    match: str,
) -> None:
    defect = {
        "category": "style_outlier",
        "severity": "noticeable",
        "target_ids": ["obj_0"],
        "scope": "visible_design_language",
        "relation": "style conflict",
        "reason": "A material style conflict is visible.",
        field: value,
    }

    with pytest.raises(ValueError, match=match):
        score_l3_metric_report(
            "style_consistency",
            {"judgement": {"verdict": "invalid", "defects": [defect]}},
            ordered_object_ids=_ids(2),
        )


@pytest.mark.parametrize(
    "field",
    ["scoring_target_ids", "causal_object_ids", "context_ids"],
)
def test_l3_scoring_rejects_unknown_audit_object_ids(field: str) -> None:
    defect = {
        "category": "style_outlier",
        "severity": "noticeable",
        "target_ids": ["obj_0"],
        "scope": "visible_design_language",
        "relation": "style conflict",
        "reason": "A material style conflict is visible.",
        field: ["unknown_object"],
    }

    with pytest.raises(ValueError, match="unknown object IDs"):
        score_l3_metric_report(
            "style_consistency",
            {"judgement": {"verdict": "invalid", "defects": [defect]}},
            ordered_object_ids=_ids(2),
        )


def test_projection_rejects_positive_burden_below_confirmed_invalid_floor() -> None:
    with pytest.raises(ValueError, match=r"minimum|0\.4|burden"):
        project_metric_events(
            "functional_consistency",
            ordered_object_ids=_ids(2),
            events=[
                {
                    "event_id": "below-minimum",
                    "burden": 0.39,
                    "allocations": {"obj_0": 0.39},
                }
            ],
        )


def test_fixed_defect_reports_denominator_sensitivity_and_floor() -> None:
    event = {
        "event_id": "one-gross-scale-defect",
        "burden": 1.0,
        "allocations": {"obj_0": 1.0},
    }
    ten_objects = project_metric_events(
        "scale_consistency",
        ordered_object_ids=_ids(10),
        events=[event],
    )
    twenty_objects = project_metric_events(
        "scale_consistency",
        ordered_object_ids=_ids(20),
        events=[event],
    )

    assert ten_objects["prevalence_deduction"] == pytest.approx(0.30)
    assert ten_objects["worst_event_floor_deduction"] == pytest.approx(0.25)
    assert ten_objects["base_metric_deduction"] == pytest.approx(0.30)
    assert ten_objects["applied_deduction_multiplier"] == pytest.approx(2.0)
    assert ten_objects["metric_deduction"] == pytest.approx(0.60)
    assert ten_objects["score"] == pytest.approx(0.40)

    assert twenty_objects["prevalence_deduction"] == pytest.approx(0.15)
    assert twenty_objects["worst_event_floor_deduction"] == pytest.approx(0.25)
    assert twenty_objects["base_metric_deduction"] == pytest.approx(0.25)
    assert twenty_objects["applied_deduction_multiplier"] == pytest.approx(2.0)
    assert twenty_objects["metric_deduction"] == pytest.approx(0.50)
    assert twenty_objects["score"] == pytest.approx(0.50)


def test_minimum_repair_set_splits_one_total_functional_burden() -> None:
    result = score_l3_metric_report(
        "functional_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "defects": [
                    {
                        "category": "approach_clearance_failure",
                        "severity": "blocked",
                        "target_ids": ["obj_0"],
                        "causal_object_ids": ["obj_1", "obj_2"],
                        "scoring_target_ids": ["obj_1", "obj_2"],
                        "attribution_mode": "minimum_repair_set",
                        "scope": "group_real_world_usability",
                        "relation": "two-object minimum repair set",
                        "reason": "Either blocker can be removed to restore access.",
                    }
                ],
            }
        },
        ordered_object_ids=_ids(4),
    )

    assert result["event_count"] == 1
    event = result["events"][0]
    assert event["burden"] == 1.0
    assert event["attribution_mode"] == "minimum_repair_set"
    assert event["allocations"] == {"obj_1": 0.5, "obj_2": 0.5}
    assert result["burden_total_b_m"] == 1.0
