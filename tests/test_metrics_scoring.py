from __future__ import annotations

from benchmark.metrics.aggregate import aggregate_case_results
from benchmark.workflow.scoring import compute_primary_score, compute_validity_gate


def test_primary_score_prompt_only_equals_room_score() -> None:
    metrics = {
        "validity_gate": True,
        "room_consistency_score_norm": 0.75,
        "object_presence_rate": None,
        "specified_relation_pass_rate": None,
        "specified_attachment_pass_rate": None,
    }

    assert compute_primary_score(metrics, "prompt_only") == 0.75


def test_primary_score_structured_basic_uses_vlm_score() -> None:
    metrics = {
        "validity_gate": True,
        "room_consistency_score_norm": 0.75,
        "object_presence_rate": 1.0,
        "specified_relation_pass_rate": None,
        "specified_attachment_pass_rate": None,
    }

    assert compute_primary_score(metrics, "structured_basic") == 0.75


def test_primary_score_structured_relation_uses_vlm_score() -> None:
    metrics = {
        "validity_gate": True,
        "room_consistency_score_norm": 0.75,
        "object_presence_rate": 1.0,
        "specified_relation_pass_rate": 0.5,
        "specified_attachment_pass_rate": 1.0,
    }

    assert compute_primary_score(metrics, "structured_relation") == 0.75


def test_validity_gate_false_forces_primary_score_zero() -> None:
    metrics = {
        "validity_gate": False,
        "room_consistency_score_norm": 1.0,
        "object_presence_rate": 1.0,
    }

    assert compute_primary_score(metrics, "structured_basic") == 0.0


def test_validity_gate_fails_layout_schema_errors() -> None:
    schema = {
        "type": "object",
        "required": ["scene_id", "objects"],
        "properties": {"scene_id": {"type": "string"}, "objects": {"type": "array"}},
    }

    result = compute_validity_gate({"input_level": "prompt_only"}, {"objects": []}, schema)

    assert not result.passed
    assert any("layout schema invalid" in error for error in result.errors)


def test_aggregate_groups_by_input_level() -> None:
    row = {
        "case_id": "case_1",
        "model": "mock",
        "input_level": "prompt_only",
        "validity_gate": True,
        "room_consistency_score": 3,
        "room_consistency_score_norm": 0.75,
        "object_presence_rate": None,
        "specified_relation_pass_rate": None,
        "specified_attachment_pass_rate": None,
        "primary_score": 0.75,
    }

    summary = aggregate_case_results([row, row])

    assert summary["num_cases"] == 2
    assert summary["by_input_level"]["prompt_only"]["primary_score_mean"] == 0.75
