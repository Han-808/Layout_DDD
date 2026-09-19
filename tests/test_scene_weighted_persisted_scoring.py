from __future__ import annotations

import pytest

from benchmark.camera_cal_scene_level.persisted_scoring import (
    run_scoring_aggregate,
)


def _room(
    case_id: str,
    *,
    score: float,
    objects: int,
) -> dict:
    return {
        "case_id": case_id,
        "n_scene": objects,
        "combined_score_100": score,
        "combined_observed_score_100": score,
        "combined_coverage_fraction": 1.0,
        "final_decision_status": "resolved",
        "metrics": [
            {
                "layer": "L1",
                "metric": "collision",
                "label": "Collision",
                "score": score / 100.0,
                "observed_score": score / 100.0,
                "coverage_fraction": 1.0,
                "overall_weight": 0.1,
            }
        ],
    }


def test_multi_room_model_score_uses_two_stage_scene_weighting() -> None:
    aggregate = run_scoring_aggregate(
        [
            _room(
                "mr.model.layout_01.room_000",
                score=0.0,
                objects=1,
            ),
            _room(
                "mr.model.layout_01.room_001",
                score=100.0,
                objects=9,
            ),
            _room(
                "mr.model.layout_02.room_000",
                score=0.0,
                objects=1,
            ),
        ]
    )

    # layout_01 = (0*1 + 100*9) / 10 = 90, weighted by 2 rooms.
    # layout_02 = 0, weighted by 1 room. Model = (90*2 + 0*1) / 3 = 60.
    assert aggregate["official_score_100"] == pytest.approx(60.0)
    assert aggregate["room_macro_diagnostic"]["official_score_100"] == (
        pytest.approx(100.0 / 3.0)
    )
    assert aggregate["scene_count"] == 2
    assert aggregate["aggregation_policy"]["within_scene"] == (
        "room_score_weighted_by_canonical_object_count"
    )
    assert aggregate["aggregation_policy"]["across_scenes"] == (
        "scene_score_weighted_by_room_count"
    )
    scenes = {item["layout_id"]: item for item in aggregate["scenes"]}
    assert scenes["layout_01"]["official_score_100"] == pytest.approx(90.0)
    assert scenes["layout_01"]["room_count"] == 2
    assert scenes["layout_01"]["object_count"] == 10
    assert scenes["layout_02"]["official_score_100"] == pytest.approx(0.0)


def test_non_multi_room_inputs_preserve_historical_room_macro_shape() -> None:
    aggregate = run_scoring_aggregate(
        [
            _room("S100", score=80.0, objects=4),
            _room("S101", score=100.0, objects=10),
        ]
    )

    assert aggregate["official_score_100"] == pytest.approx(90.0)
    assert "aggregation_policy" not in aggregate
    assert "room_macro_diagnostic" not in aggregate
    assert "scenes" not in aggregate
