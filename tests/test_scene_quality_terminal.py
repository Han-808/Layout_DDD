from __future__ import annotations

import pytest

from benchmark.evaluator.scene_quality.terminal import (
    infrastructure_failure_from_scope,
    recoverable_validation_failure,
    terminalize_required_scope,
)
from benchmark.visual_judge.contracts import ResponseSchemaRepairError


def test_non_hard_unresolved_scope_defaults_binary_with_ambiguity() -> None:
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

    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated_degraded"
    assert record["score"] == 1.0
    assert record["judgement"]["verdict"] == "valid"
    assert record["judgement"]["evidence_ambiguous"] is True
    assert record["judgement"]["defaulted"] is True


@pytest.mark.parametrize(
    "stop_reason",
    [
        "camera_candidate_bank_failed",
        "camera_constraint_contract_invalid",
        "camera_selector_failed",
        "camera_selector_unavailable",
        "render_failed",
    ],
)
def test_acquisition_stop_defaults_binary_when_prior_evidence_is_retained(
    stop_reason: str,
) -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "insufficient_visual_evidence",
            "judgement": {"verdict": "ambiguous"},
            "camera_control_audit": {
                "stop_reason": stop_reason,
                "audit": {
                    "images_used": ["path:/tmp/retained.png"],
                    "trace": [
                        {
                            "stage": "evidence_gate",
                            "result": {"ready": True},
                            "images_used": ["path:/tmp/retained.png"],
                        }
                    ],
                },
            },
        },
        phase="group_local:group_002",
    )

    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated_degraded"
    assert record["terminal_decision"]["defaulted"] is True
    assert record["terminal_decision"]["failure"][
        "controller_stop_reason"
    ] == stop_reason


@pytest.mark.parametrize(
    "stop_reason",
    [
        "camera_candidate_bank_failed",
        "camera_constraint_contract_invalid",
        "camera_selector_failed",
        "camera_selector_unavailable",
        "render_failed",
    ],
)
def test_acquisition_failure_without_explicit_gate_ready_packet_is_hard(
    stop_reason: str,
) -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "camera acquisition failed",
            "evidence_paths": ["/tmp/not-gate-validated.png"],
            "camera_control_audit": {
                "stop_reason": stop_reason,
                "audit": {"images_used": ["path:/tmp/not-gate-validated.png"]},
            },
        },
        phase="group_local:group_acquisition",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


def test_renderer_followup_contract_violation_is_unconditionally_hard() -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "renderer exceeded its validated authorization",
            "camera_control_audit": {
                "stop_reason": "renderer_followup_contract_invalid",
                "audit": {
                    "images_used": ["path:/tmp/retained.png"],
                    "trace": [
                        {
                            "stage": "evidence_gate",
                            "result": {"ready": True},
                            "images_used": ["path:/tmp/retained.png"],
                        }
                    ],
                },
            },
        },
        phase="group_local:renderer_contract",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


def test_evaluated_status_cannot_bypass_unknown_controller_stop_proof() -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "score": 1.0,
            "judgement": {"verdict": "valid"},
            "camera_control_audit": {
                "stop_reason": "unknown_acquisition_stop",
                "audit": {"images_used": [], "trace": []},
            },
        },
        phase="group_local:evaluated_without_proof",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


def test_unknown_controller_stop_is_hard_even_with_gate_ready_evidence() -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "camera_control_audit": {
                "stop_reason": "unknown_acquisition_stop",
                "audit": {
                    "images_used": ["path:/tmp/retained.png"],
                    "trace": [
                        {
                            "stage": "evidence_gate",
                            "result": {"ready": True},
                            "images_used": ["path:/tmp/retained.png"],
                        }
                    ],
                },
            },
        },
        phase="group_local:unknown_controller_stop",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


def test_successful_controller_stop_does_not_require_failure_proof() -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "score": 1.0,
            "judgement": {"verdict": "valid"},
            "camera_control_audit": {
                "stop_reason": "judge_conclusion",
                "audit": {"images_used": [], "trace": []},
            },
        },
        phase="group_local:judge_conclusion",
    )

    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated"


@pytest.mark.parametrize(
    "stop_reason",
    [
        "blank_evidence",
        "corrupt_evidence",
        "undecodable_evidence",
        "evidence_missing",
        "manifest_failure",
    ],
)
def test_integrity_stop_reason_is_hard_even_with_an_earlier_ready_packet(
    stop_reason: str,
) -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "visual evidence is not technically ready",
            "camera_control_audit": {
                "stop_reason": stop_reason,
                "audit": {
                    "images_used": ["path:/tmp/retained.png"],
                    "trace": [
                        {
                            "stage": "evidence_gate",
                            "result": {"ready": True},
                            "images_used": ["path:/tmp/retained.png"],
                        }
                    ],
                },
            },
        },
        phase="group_local:group_integrity",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


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


@pytest.mark.parametrize(
    "flag",
    ["defaulted", "forced_binary", "evidence_ambiguous"],
)
def test_explicit_degraded_judgement_flag_marks_evaluated_scope_degraded(
    flag: str,
) -> None:
    record = terminalize_required_scope(
        {
            "status": "evaluated",
            "score": 1.0,
            "judgement": {"verdict": "valid", flag: True},
        },
        phase="group_local:group_flagged",
    )

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


def test_schema_failure_defaults_only_the_affected_scope() -> None:
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

    assert record["status"] == "evaluated"
    assert record["terminal_state"] == "evaluated_degraded"
    assert record["judgement"]["defaulted"] is True
    assert infrastructure_failure_from_scope(
        record,
        phase="group_local",
        scope_id="group_004",
    ) is None


def test_transport_failure_remains_hard_and_keeps_scope_identity() -> None:
    record = terminalize_required_scope(
        {
            "status": "failed",
            "reason": "vlm_judge_failed",
            "judgement": {
                "error_type": "EndpointHTTPError",
                "error": "HTTP 429",
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
        "error_type": "EndpointHTTPError",
        "error": "HTTP 429",
    }


def test_missing_all_render_evidence_remains_hard() -> None:
    record = terminalize_required_scope(
        {
            "status": "unresolved",
            "reason": "group_local_render_evidence_unavailable",
        },
        phase="group_local:group_005",
    )

    assert record["status"] == "failed"
    assert record["terminal_state"] == "infrastructure_failure"


def test_recoverable_validation_classifier_rejects_program_and_transport_errors(
) -> None:
    assert recoverable_validation_failure(ValueError("invalid schema")) is True
    assert recoverable_validation_failure(TypeError("program bug")) is False
    assert recoverable_validation_failure(KeyError("program bug")) is False
    assert recoverable_validation_failure(
        ResponseSchemaRepairError(
            "schema remained invalid",
            schema_audit={"attempts": [{"attempt": 2}]},
        )
    ) is True
    assert recoverable_validation_failure(
        ResponseSchemaRepairError(
            "repair transport failed",
            schema_audit={
                "attempts": [
                    {"attempt": 2, "failure_kind": "transport"}
                ]
            },
        )
    ) is False
