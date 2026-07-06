from __future__ import annotations

from benchmark.metrics.metrics import compute_case_metrics


def _entry(iteration: int, flags: list[dict], *, feedback: dict | None = None) -> dict:
    return {
        "iteration": iteration,
        "evaluation": {
            "metrics": {
                "case_id": "case_1",
                "model": "mock",
                "input_level": "structured_basic",
                "validity_gate": True,
            },
            "debug_evidence": {"physical_flags": flags},
        },
        "feedback": feedback or {},
    }


def test_case_metrics_include_physical_repair_deltas() -> None:
    initial = [
        {"type": "serious_collision", "objects": ["a", "b"], "confidence": "high"},
        {"type": "room_boundary", "objects": ["c"], "confidence": "low", "source_kind": "object_position_extent_fallback"},
    ]
    final = [{"type": "room_boundary", "objects": ["c"], "confidence": "low", "source_kind": "object_position_extent_fallback"}]
    feedback = {
        "repair_actions": [
            {"action": "spread_dense_collision_cluster", "objects": ["a", "b", "c", "d"]},
        ]
    }

    metrics = compute_case_metrics([_entry(0, initial, feedback=feedback), _entry(1, final)])

    assert metrics["serious_collision_count_initial"] == 1
    assert metrics["serious_collision_count_final"] == 0
    assert metrics["serious_collision_delta"] == -1
    assert metrics["room_boundary_count_initial"] == 1
    assert metrics["room_boundary_count_final"] == 1
    assert metrics["room_boundary_delta"] == 0
    assert metrics["boundary_count_initial"] == 1
    assert metrics["boundary_count_final"] == 1
    assert metrics["repair_helped_physical_flags"] is True
    assert metrics["low_confidence_physical_flag_count"] == 1
    assert metrics["fallback_physical_flag_count"] == 1
    assert metrics["dense_collision_cluster_count"] == 1
    assert metrics["dense_collision_cluster_max_size"] == 4


def test_case_metrics_count_fallback_metadata_conflict_codes() -> None:
    final = [
        {
            "type": "impossible_height_constraint",
            "code": "fallback_metadata_conflict",
            "objects": ["too_tall"],
            "confidence": "low",
            "source_kind": "fallback_default",
        }
    ]

    metrics = compute_case_metrics([_entry(0, []), _entry(1, final)])

    assert metrics["fallback_metadata_conflict_count"] == 1
    assert metrics["fallback_physical_flag_count"] == 1
