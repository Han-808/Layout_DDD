from __future__ import annotations

from benchmark.evaluator.scene_quality.terminal import (
    infrastructure_failure_from_scope,
    terminalize_required_scope,
)


def test_final_scientific_scope_cannot_remain_unresolved() -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "insufficient_visual_evidence",
            "judgement": {
                "verdict": "ambiguous",
                "evidence_status": "insufficient",
            },
        },
        phase="group_local:group_001",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"
    assert record["reason"] == "insufficient_visual_evidence"
    assert record["infrastructure_failure"]["failure_kind"] == (
        "terminal_contract_failure"
    )
    assert record["infrastructure_failure"]["original_status"] == (
        "unresolved"
    )


def test_selector_failure_is_an_engineering_failure_without_binary_result(
) -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "insufficient_visual_evidence",
            "judgement": {"verdict": "ambiguous"},
            "camera_control_audit": {
                "stop_reason": "camera_selector_failed"
            },
        },
        phase="group_local:group_002",
    )

    assert record["status"] == "failed"
    assert record["infrastructure_failure"]["failure_kind"] == (
        "engineering_failure"
    )
    assert record["infrastructure_failure"][
        "controller_stop_reason"
    ] == "camera_selector_failed"


def test_forced_binary_scope_is_evaluated_degraded() -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "score": 1.0,
            "judgement": {
                "verdict": "valid",
                "budget_exhaustion_forced_choice": {
                    "applied": True,
                    "trigger": "max_total_images_exhausted",
                },
            },
        },
        phase="group_local:group_003",
    )

    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated_degraded"
    assert record["degradation_audit"]["forced_binary"] is True


def test_reterminalization_preserves_existing_degraded_state() -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "terminal_state": "evaluated_degraded",
            "score": 1.0,
            "judgement": {"verdict": "valid"},
        },
        phase="scene_global",
    )

    assert record["terminal_state"] == "evaluated_degraded"
    assert record["degradation_audit"]["forced_binary"] is False


def test_normal_binary_scope_is_evaluated() -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "score": 0.0,
            "judgement": {"verdict": "invalid"},
        },
        phase="scene_global",
    )

    assert record["terminal_state"] == "evaluated"
    assert infrastructure_failure_from_scope(
        record,
        phase="scene_global",
        scope_id="scene_global",
    ) is None


def test_failure_projection_keeps_scope_identity() -> None:
    record = terminalize_required_scope(
        {
            "status": "failed",
            "reason": "vlm_judge_failed",
            "judgement": {
                "error_type": "ResponseSchemaRepairError",
                "error": "response remained invalid",
            },
        },
        phase="group_local:group_004",
    )

    failure = infrastructure_failure_from_scope(
        record,
        phase="group_local",
        scope_id="group_004",
    )
    assert failure == {
        "phase": "group_local",
        "scope_id": "group_004",
        "failure_kind": "engineering_failure",
        "reason": "vlm_judge_failed",
        "controller_stop_reason": None,
        "error_type": "ResponseSchemaRepairError",
        "error": "response remained invalid",
    }
