"""Pure accounting tests; no models, assets, renderer or network."""
from copy import deepcopy

import pytest

from benchmark.visual_judge.evidence_gap_v2 import coverage_from_plan, gap_record


def complete(unit_id, *, visual=True):
    return {
        "unit_id": unit_id, "accepted": True, "status": "evaluated",
        "score": 0.8, "execution_complete": True,
        "visual_observation_complete": visual,
    }


def test_gap_does_not_complete_evaluation_or_invent_score():
    known = complete("one")
    gap = gap_record(unit_id="two", reason="bounded_camera_bank_empty", source="camera")
    snapshot = deepcopy([known, gap])
    coverage = coverage_from_plan(["one", "two", "three"], [known, gap])
    assert coverage["planned_count"] == 3
    assert coverage["evaluation_count"] == 1
    assert coverage["evaluation_fraction"] == pytest.approx(1 / 3)
    assert coverage["visual_evidence_fraction"] == pytest.approx(1 / 3)
    assert coverage["missing_ids"] == ["three"]
    assert coverage["gap_count"] == 1 and coverage["failure_count"] == 0
    assert not coverage["execution_complete"]
    assert not coverage["evaluation_complete"]
    assert gap["score"] is None and gap["verdict"] == "unknown"
    assert [known, gap] == snapshot


def test_execution_can_finish_with_gap_without_complete_evaluation():
    gap = gap_record(unit_id="one", reason="unobserved_surface", source="retained_visuals")
    coverage = coverage_from_plan(["one"], [gap])
    assert coverage["execution_complete"]
    assert not coverage["evaluation_complete"]
    assert not coverage["visual_evidence_complete"]


def test_existing_geometry_rule_score_does_not_claim_visual_coverage():
    record = complete("geometry", visual=False)
    record["decision_source"] = "geometry_only"
    record["rule_id"] = "metric_owned_verified_rule"
    coverage = coverage_from_plan(["geometry"], [record])
    assert coverage["evaluation_complete"] and coverage["execution_complete"]
    assert coverage["visual_evidence_fraction"] == 0
    assert not coverage["visual_evidence_complete"]
    assert coverage["units"][0]["score"] == 0.8


def test_infrastructure_failure_is_not_a_coverage_gap():
    failure = {
        "unit_id": "one", "status": "failed", "score": None,
        "accepted": False, "execution_complete": True,
        "failure_category": "model_service_failure",
    }
    coverage = coverage_from_plan(["one"], [failure])
    assert coverage["failure_count"] == 1 and coverage["gap_count"] == 0
    assert coverage["failures"] == [failure]


@pytest.mark.parametrize("planned,records", [
    (["one", "one"], []),
    ([""], []),
    (["one"], [complete("other")]),
    (["one"], [complete("one"), complete("one")]),
    (["one"], [{**complete("one"), "score": None}]),
    (["one"], [{**complete("one"), "score": float("nan")}]),
    (["one"], [{**complete("one"), "score": True}]),
    (["one"], [{**complete("one"), "status": "failed"}]),
    (["one"], [{**complete("one"), "failure_category": "model_service_failure"}]),
    (["one"], [{"unit_id": "one", "status": "not_evaluable", "score": 0}]),
])
def test_invalid_accounting_is_rejected(planned, records):
    with pytest.raises(ValueError):
        coverage_from_plan(planned, records)


def test_empty_plan_is_not_evaluation_or_visual_coverage_claim():
    coverage = coverage_from_plan([], [])
    assert coverage["evaluation_fraction"] is None
    assert coverage["visual_evidence_fraction"] is None
    assert not coverage["evaluation_complete"]
    assert not coverage["visual_evidence_complete"]


def test_gap_copies_evidence_and_requires_reason():
    evidence = {"paths": ["retained"]}
    gap = gap_record(unit_id="one", reason="insufficient", source="camera", evidence=evidence)
    evidence["paths"].append("later")
    assert gap["evidence"] == {"paths": ["retained"]}
    with pytest.raises(ValueError):
        gap_record(unit_id="one", reason="", source="camera")
