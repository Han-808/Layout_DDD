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
