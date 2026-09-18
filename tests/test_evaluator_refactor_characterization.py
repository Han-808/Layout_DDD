"""Characterization tests pinning behavior preserved by the v8 refactor.

Each test records the exact pre-refactor behavior of a helper that the
refactor moves or that has an intentionally divergent sibling. These are
regression anchors, not new requirements.
"""

from __future__ import annotations

from benchmark.evaluator.scene_quality import cross_group_relations
from benchmark.evaluator.scene_quality import functional_acquisition
from benchmark.evaluator.scene_quality import functional_checks
from benchmark.evaluator.scene_quality import placement_checks


class TestFunctionalStableUnique:
    """repr-marker dedup shared by the functional/cross-group ledgers."""

    def test_dedup_uses_repr_and_keeps_first_occurrence(self) -> None:
        first = {"a": 1}
        second = {"a": 1}
        third = {"b": 2}
        result = functional_checks._stable_unique([first, second, third])
        assert result == [{"a": 1}, {"b": 2}]
        # No copying: the retained entry is the original object.
        assert result[0] is first
        assert result[1] is third

    def test_nan_values_collapse_by_repr_despite_inequality(self) -> None:
        nan = float("nan")
        other_nan = float("nan")
        assert nan != other_nan
        result = functional_checks._stable_unique([nan, other_nan])
        assert len(result) == 1

    def test_all_three_modules_expose_the_same_function(self) -> None:
        # cross_group_relations and functional_acquisition use the same
        # dedup semantics as functional_checks.
        values = [{"a": 1}, {"a": 1}, "x", "x", 3]
        expected = [{"a": 1}, "x", 3]
        assert functional_checks._stable_unique(values) == expected
        assert cross_group_relations._stable_unique(values) == expected
        assert functional_acquisition._stable_unique(values) == expected


class TestPlacementStableUniqueIsIntentionallyDifferent:
    """The placement ledger variant dedups by equality and deep-copies."""

    def test_dedup_uses_equality_and_returns_copies(self) -> None:
        first = {"a": [1]}
        second = {"a": [1]}
        result = placement_checks._stable_unique([first, second])
        assert result == [{"a": [1]}]
        # Deep copy: the retained entry is NOT the original object.
        assert result[0] is not first
        assert result[0]["a"] is not first["a"]

    def test_nan_values_are_retained_because_nan_is_unequal(self) -> None:
        nan = float("nan")
        other_nan = float("nan")
        result = placement_checks._stable_unique([nan, other_nan])
        # Divergence from the functional variant: equality-based dedup
        # keeps both NaN entries where repr-based dedup keeps one.
        assert len(result) == 2


class TestAppendObligationTransition:
    """Lifecycle append shared verbatim by functional and placement ledgers."""

    def test_appends_and_coerces_to_str(self) -> None:
        check: dict = {}
        functional_checks._append_obligation_transition(
            check, "scheduled", source=1
        )
        assert check["obligation_lifecycle"] == [
            {"state": "scheduled", "source": "1"}
        ]

    def test_consecutive_duplicate_transition_is_not_appended(self) -> None:
        check: dict = {}
        for _ in range(2):
            functional_checks._append_obligation_transition(
                check, "judged", source="phase"
            )
        assert check["obligation_lifecycle"] == [
            {"state": "judged", "source": "phase"}
        ]

    def test_non_adjacent_duplicate_is_appended_again(self) -> None:
        check: dict = {}
        functional_checks._append_obligation_transition(
            check, "scheduled", source="a"
        )
        functional_checks._append_obligation_transition(
            check, "judged", source="a"
        )
        functional_checks._append_obligation_transition(
            check, "scheduled", source="a"
        )
        assert [t["state"] for t in check["obligation_lifecycle"]] == [
            "scheduled",
            "judged",
            "scheduled",
        ]

    def test_placement_module_matches_functional_behavior(self) -> None:
        left: dict = {}
        right: dict = {}
        functional_checks._append_obligation_transition(
            left, "deferred_budget", source="routing"
        )
        placement_checks._append_obligation_transition(
            right, "deferred_budget", source="routing"
        )
        assert left == right


# ---------------------------------------------------------------------------
# case_scoring_summary characterization (step 2 of the v8 refactor)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from benchmark.camera_cal_scene_level.persisted_scoring import (
    case_scoring_summary,
)

_PERSISTED_SNAPSHOT = (
    Path(__file__).parent
    / "fixtures"
    / "persisted_scoring_characterization_v1.json"
)

_PERSISTED_SCENARIOS = {
    "empty_inputs": dict(
        case_id="case_empty",
        case_manifest={},
        l1_report={},
        l3_report={},
    ),
    "partial_l3_only": dict(
        case_id="case_partial",
        case_manifest={
            "scoring_profile": {
                "scoring_profile_id": "intrinsic_validity_v2",
                "scoring_spec_version": "object_equivalent_burden_v3",
                "deduction_multiplier": 2,
                "layer_weights": {
                    "l1_physical_plausibility": 0.3,
                    "l3_scene_quality": 0.7,
                },
            },
            "canonical_object_denominator": {
                "n_scene": 2,
                "ordered_object_ids": ["obj_a", "obj_b"],
            },
            "final_decision_status": "unresolved",
        },
        l1_report={"status": "complete", "metrics": {}},
        l3_report={
            "status": "complete",
            "scoring": {
                "metric_weights": {
                    "functional_consistency": 0.52,
                    "semantic_placement_consistency": 0.28,
                }
            },
            "metrics": {
                "functional_consistency": {
                    "status": "evaluated",
                    "score": 0.9,
                    "judgement": {"verdict": "valid", "reason": "ok"},
                    "coverage": {
                        "fraction": 0.5,
                        "complete": False,
                        "score_grounding": {
                            "fraction": 0.9,
                            "complete": False,
                        },
                    },
                    "scoring": {
                        "coefficient_n_m": 2.0,
                        "burden_total_b_m": 0.4,
                        "p_max": 0.4,
                        "metric_deduction": 0.1,
                        "event_count": 1,
                        "events": [
                            {"event_id": "evt", "burden": 0.4},
                            "not-a-dict",
                        ],
                    },
                },
                "semantic_placement_consistency": {
                    "status": "evaluated",
                    "judgement": {"verdict": None},
                    "coverage": {},
                    "scoring": {
                        "coverage_projection": {
                            "raw_score_before_coverage_projection": 0.8,
                        },
                        "placement_component_weights": {
                            "typed": 0.8,
                            "residual_global_review": 0.2,
                        },
                        "placement_components": {
                            "typed": {
                                "score": 0.75,
                                "metric_deduction": 0.25,
                                "event_count": 2,
                            },
                            "bad": "not-a-dict",
                        },
                    },
                },
            },
        },
    ),
    "engineering_failure_dedup": dict(
        case_id="case_failures",
        case_manifest={
            "scoring_reliability": {"summary": "kept"},
        },
        l1_report={
            "backend_report": {
                "scoring": {"metric_weights": {"collision": 0.5}},
            },
        },
        l3_report={},
        l1_diagnostics={
            "engineering_failures": [
                {"metric": "collision", "error": "boom"},
                {"metric": "collision", "error": "boom"},
                {"metric": "collision", "route": "r2"},
                {"error": "boom"},
                "not-a-dict",
            ]
        },
    ),
}


class TestCaseScoringSummaryCharacterization:
    """Snapshot of pre-refactor outputs for boundary-shaped inputs."""

    def test_outputs_match_recorded_snapshot(self) -> None:
        expected = json.loads(_PERSISTED_SNAPSHOT.read_text())
        assert set(expected) == set(_PERSISTED_SCENARIOS)
        for name, kwargs in _PERSISTED_SCENARIOS.items():
            actual = json.loads(json.dumps(case_scoring_summary(**kwargs)))
            assert actual == expected[name], name
