from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark.camera_cal_scene_level.leaderboard_scoring import (
    load_leaderboard_scoring_profile,
    rescore_scene_generation_case,
    validate_leaderboard_scoring_profile,
)
from benchmark.resources import packaged_resource_path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_RELATIVE = (
    "configs/evaluation/scene_generation_leaderboard_scoring_v1.json"
)
PROFILE_PATH = ROOT / PROFILE_RELATIVE


def _reports() -> tuple[dict, dict]:
    l1 = {
        "metrics": {
            "collision": {"scoring": {"base_metric_deduction": 0.10}},
            "oob": {"scoring": {"base_metric_deduction": 0.05}},
            "support": {"scoring": {"base_metric_deduction": 0.20}},
        }
    }
    l3 = {
        "metrics": {
            "style_consistency": {
                "scoring": {"base_metric_deduction": 0.02}
            },
            "scale_consistency": {
                "scoring": {"base_metric_deduction": 0.10}
            },
            "object_pairing_consistency": {
                "scoring": {"base_metric_deduction": 0.05}
            },
            "functional_consistency": {
                "scoring": {"base_metric_deduction": 0.10}
            },
            "semantic_placement_consistency": {
                "scoring": {
                    "placement_component_weights": {
                        "local": 0.6,
                        "global": 0.4,
                    },
                    "placement_components": {
                        "local": {"base_metric_deduction": 0.20},
                        "global": {"base_metric_deduction": 0.10},
                    },
                }
            },
        }
    }
    return l1, l3


def test_web_profile_is_packaged_byte_for_byte_and_has_current_weights() -> None:
    profile = load_leaderboard_scoring_profile()

    assert packaged_resource_path(PROFILE_RELATIVE).read_bytes() == (
        PROFILE_PATH.read_bytes()
    )
    assert profile["category_weights"] == {
        "physical_plausibility": 0.36,
        "functional_semantics": 0.50,
        "visual_coherence": 0.14,
    }
    assert profile["metric_weights"] == {
        "collision": 0.14,
        "oob": 0.08,
        "support": 0.14,
        "style_consistency": 0.03,
        "scale_consistency": 0.03,
        "object_pairing_consistency": 0.08,
        "functional_consistency": 0.30,
        "semantic_placement_consistency": 0.20,
    }
    assert profile["deduction_multipliers"] == {
        "collision": 2.4,
        "oob": 3.0,
        "support": 2.05,
        "style_consistency": 3.0,
        "scale_consistency": 3.0,
        "object_pairing_consistency": 3.0,
        "functional_consistency": 1.5,
        "semantic_placement_consistency": 1.75,
    }
    assert len(profile["profile_sha256"]) == 64


def test_posthoc_formula_matches_the_current_web_projection() -> None:
    l1, l3 = _reports()
    original = deepcopy((l1, l3))

    result = rescore_scene_generation_case(l1_report=l1, l3_report=l3)

    assert result is not None
    assert result["profile_id"] == "scene_generation_leaderboard_web_v1"
    assert result["fine_metrics_100"] == pytest.approx(
        {
            "collision": 76.0,
            "oob": 85.0,
            "support": 59.0,
            "style_consistency": 94.0,
            "scale_consistency": 70.0,
            "object_pairing_consistency": 85.0,
            "functional_consistency": 85.0,
            "semantic_placement_consistency": 72.0,
        }
    )
    assert result["physical_plausibility_100"] == pytest.approx(
        71.38888888888889
    )
    assert result["functional_semantics_100"] == pytest.approx(79.8)
    assert result["visual_coherence_100"] == pytest.approx(
        83.71428571428571
    )
    assert result["overall_100"] == pytest.approx(77.32)
    assert (l1, l3) == original


def test_legacy_score_fallback_preserves_the_web_formula() -> None:
    l1, l3 = _reports()
    l1["metrics"]["collision"] = {"score": 0.8}

    result = rescore_scene_generation_case(l1_report=l1, l3_report=l3)

    assert result is not None
    assert result["fine_metrics_100"]["collision"] == pytest.approx(76.0)


def test_profile_validation_rejects_weight_drift() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile["metric_weights"]["collision"] = 0.15

    with pytest.raises(ValueError, match="sum to 1.0|group weight mismatch"):
        validate_leaderboard_scoring_profile(profile)


def test_posthoc_projection_requires_all_three_categories() -> None:
    l1, l3 = _reports()
    del l3["metrics"]["functional_consistency"]
    del l3["metrics"]["semantic_placement_consistency"]

    assert rescore_scene_generation_case(l1_report=l1, l3_report=l3) is None
