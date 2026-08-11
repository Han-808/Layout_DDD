from __future__ import annotations

from benchmark.evaluator.scene_quality.functional_group_evidence import (
    FunctionalGroupEvidenceBank,
)
from benchmark.evaluator.scene_quality.group_scoped import (
    evaluate_group_scoped_judgements,
)
from benchmark.visual_judge.group_scope import GroupCameraScope
from benchmark.visual_judge.orchestration.evidence_window import (
    compose_bounded_evidence_window,
    resolve_bounded_evidence_window,
    select_reusable_evidence,
)


def _packet() -> dict:
    return {
        "group": {
            "group_id": "group_001",
            "object_ids": ["chair", "table", "lamp"],
        },
        "paths": ["global.png", "group-local.png", "chair-probe.png"],
        "resolution": {
            "scope_satisfied": True,
            "functional_probe_reuse": {
                "baseline_packet_paths": [
                    "global.png",
                    "group-local.png",
                ],
                "requested_probe_paths": [
                    "chair-probe.png",
                    "lamp-probe.png",
                ],
            },
        },
        "functional_probe_evidence": {
            "image_order": [
                {
                    "artifact_id": "chair-probe.png",
                    "check_ids": ["check-chair"],
                    "target_ids": ["chair"],
                    "required_observations": ["interaction_side_visible"],
                },
                {
                    "artifact_id": "lamp-probe.png",
                    "check_ids": ["check-lamp"],
                    "target_ids": ["lamp"],
                    "required_observations": ["target_visible"],
                },
            ]
        },
    }


def test_group_bank_keeps_two_fixed_images_and_selects_relevant_probes() -> None:
    bank = FunctionalGroupEvidenceBank.from_packet(
        _packet(),
        max_active_images=6,
    )

    evidence, event = bank.initial_window(
        {
            "check_id": "check-chair",
            "target_ids": ["chair", "table"],
            "required_observations": ["interaction_side_visible"],
        }
    )

    assert evidence == [
        "global.png",
        "group-local.png",
        "chair-probe.png",
    ]
    assert event["fixed_artifact_ids"] == [
        "path:global.png",
        "path:group-local.png",
    ]
    assert event["selected_reusable_artifact_ids"] == [
        "path:chair-probe.png"
    ]
    serialized = bank.to_dict()
    lamp = next(
        item
        for item in serialized["artifacts"]
        if item["artifact_id"] == "path:lamp-probe.png"
    )
    assert lamp["consumer_check_ids"] == []


def test_controller_render_is_absorbed_and_reused_by_a_later_check() -> None:
    bank = FunctionalGroupEvidenceBank.from_packet(
        _packet(),
        max_active_images=6,
    )
    first_check = {
        "check_id": "check-chair",
        "target_ids": ["chair", "table"],
        "required_observations": ["joint_visibility"],
    }
    bank.absorb_controller_audit(
        {
            "audit": {
                "evidence_window": {
                    "initial_artifact_ids": [
                        "path:global.png",
                        "path:group-local.png",
                    ],
                    "final_artifact_ids": [
                        "path:global.png",
                        "path:group-local.png",
                        "path:chair-table-detail.png",
                    ],
                    "events": [],
                },
                "trace": [
                    {
                        "stage": "render",
                        "status": "completed",
                        "evidence_round": 1,
                        "selection_stage": "deterministic",
                        "result": {
                            "visual_evidence": [
                                "chair-table-detail.png"
                            ],
                            "provenance": {"camera_id": "detail-1"},
                        },
                    }
                ],
            }
        },
        check=first_check,
    )

    evidence, event = bank.initial_window(
        {
            "check_id": "check-table",
            "target_ids": ["table"],
            "required_observations": ["joint_visibility"],
        }
    )

    assert "chair-table-detail.png" in evidence
    assert "path:chair-table-detail.png" in event[
        "selected_reusable_artifact_ids"
    ]
    artifact = next(
        item
        for item in bank.to_dict()["artifacts"]
        if item["artifact_id"] == "path:chair-table-detail.png"
    )
    assert artifact["sources"][0]["source_check_id"] == "check-chair"
    assert artifact["consumer_check_ids"] == ["check-table"]


def test_bank_search_does_not_reuse_observation_only_evidence() -> None:
    window = resolve_bounded_evidence_window(
        {
            "functional_group_evidence_window": {
                "policy": "shared_group_bank",
                "group_id": "group_001",
                "check_id": "check-chair",
                "max_active_images": 6,
                "fixed_artifact_ids": [
                    "path:global.png",
                    "path:group-local.png",
                ],
                "reusable_artifacts": [
                    {
                        "artifact_id": "path:lamp-detail.png",
                        "visual_evidence": "lamp-detail.png",
                        "check_ids": ["check-lamp"],
                        "target_ids": ["lamp"],
                        "required_observations": ["joint_visibility"],
                    }
                ],
            }
        },
        initial_evidence=["global.png", "group-local.png"],
    )
    assert window is not None

    selected = select_reusable_evidence(
        window,
        active_evidence=["global.png", "group-local.png"],
        target_ids=["chair"],
        missing_observations=["joint_visibility"],
    )

    assert selected == []


def test_active_window_overflow_evicts_only_dynamic_evidence() -> None:
    previous = [
        "global.png",
        "group-local.png",
        "old-1.png",
        "old-2.png",
        "old-3.png",
        "old-4.png",
    ]
    window = resolve_bounded_evidence_window(
        {
            "functional_group_evidence_window": {
                "policy": "shared_group_bank",
                "group_id": "group_001",
                "check_id": "check-chair",
                "max_active_images": 6,
                "fixed_artifact_ids": [
                    "path:global.png",
                    "path:group-local.png",
                ],
                "reusable_artifacts": [],
            }
        },
        initial_evidence=previous,
    )
    assert window is not None

    current, event = compose_bounded_evidence_window(
        window,
        previous=previous,
        additions=["new-1.png", "new-2.png"],
        trigger="camera_render",
    )

    assert current == [
        "global.png",
        "group-local.png",
        "new-1.png",
        "new-2.png",
    ]
    assert event["overflow_flush_applied"] is True
    assert event["evicted_artifact_ids"] == [
        "path:old-1.png",
        "path:old-2.png",
        "path:old-3.png",
        "path:old-4.png",
    ]
    assert event["physical_artifacts_deleted"] is False


def _required_check(
    check_id: str,
    target_ids: list[str],
) -> dict:
    return {
        "check_id": check_id,
        "check_type": "relative_use_geometry",
        "check_family": "within_group_correspondence",
        "owner_stage": "group_local",
        "route_scope": "group_local",
        "target_ids": target_ids,
        "group_ids": ["group_001"],
        "owning_group_id": "group_001",
        "relation": "relative_use_geometry",
        "predicate": "relative_use_geometry",
        "required_observations": [
            "target_visible",
            "joint_visibility",
        ],
    }


def _group_packet(*, paths: list[str]) -> dict:
    checks = [
        _required_check("check-a-b", ["a", "b"]),
        _required_check("check-b-c", ["b", "c"]),
    ]
    scope = GroupCameraScope(
        group_id="group_001",
        member_ids=("a", "b", "c"),
        target_bounds_min=(0.0, 0.0, 0.0),
        target_bounds_max=(3.0, 3.0, 3.0),
        focus_center=(1.5, 1.5, 1.5),
        extent=(3.0, 3.0, 3.0),
        required_observations=("joint_visibility",),
        require_global_anchor=True,
    )
    return {
        "group": {
            "group_id": "group_001",
            "object_ids": ["a", "b", "c"],
        },
        "group_scope": scope,
        "paths": paths,
        "resolution": {
            "scope_satisfied": True,
            "provider_status": "success",
            "functional_probe_reuse": {
                "baseline_packet_paths": list(paths),
                "requested_probe_paths": [],
            },
        },
        "functional_probe_evidence": {
            "required_checks": checks,
            "required_check_ids": [
                "check-a-b",
                "check-b-c",
            ],
            "required_check_count": 2,
            "image_order": [],
        },
    }


def _evaluate_shared_packet(
    packet: dict,
    *,
    call_judge,
    audit_records: list[dict],
) -> dict:
    class Judge:
        pass

    judge = Judge()
    judge.audit_records = audit_records
    return evaluate_group_scoped_judgements(
        base={
            "status": "pending",
            "score": None,
            "reason": None,
            "evidence_request": {
                "vlm_invoked": False,
                "renderer_invoked": False,
            },
        },
        metric_name="functional_consistency",
        scene={
            "objects": [
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
            ]
        },
        prompt=None,
        packets=[packet],
        vlm_judge=judge,
        authorized_deviations=[],
        visual_style_spec=None,
        build_judge_request=lambda **kwargs: kwargs,
        call_judge=call_judge,
        apply_prompt_exemptions=lambda value, **_: value,
        normalize_judgement=lambda value, **_: {
            "status": "evaluated",
            "score": 0.0 if value["verdict"] == "invalid" else 1.0,
            "reason": None,
        },
        evidence_phase="group_local_review",
        decision_mode="final",
        group_local_check_granularity="per_check",
        group_local_evidence_policy="shared_group_bank",
        group_local_active_window_max_images=6,
    )


def test_shared_scheduler_promotes_first_check_render_for_second_check() -> None:
    calls: list[dict] = []
    audit_records: list[dict] = []

    def call_judge(judge, request: dict) -> dict:
        calls.append(request)
        check = request["functional_probe_evidence"]["required_checks"][0]
        check_id = check["check_id"]
        initial_paths = list(request["render_evidence"])
        rendered = ["shared-a-b-detail.png"] if check_id == "check-a-b" else []
        final_paths = list(dict.fromkeys([*initial_paths, *rendered]))
        judge.audit_records.append(
            {
                "audit": {
                    "evidence_window": {
                        "initial_artifact_ids": [
                            f"path:{path}" for path in initial_paths
                        ],
                        "final_artifact_ids": [
                            f"path:{path}" for path in final_paths
                        ],
                        "events": [],
                    },
                    "trace": [
                        {
                            "stage": "render",
                            "status": "completed",
                            "evidence_round": 1,
                            "selection_stage": "deterministic",
                            "result": {
                                "visual_evidence": rendered,
                                "provenance": {},
                            },
                        }
                    ]
                    if rendered
                    else [],
                }
            }
        )
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 0.9,
            "reason": "The isolated required check is satisfied.",
            "missing_evidence": [],
            "defects": [],
            "evidence_request": None,
            "functional_check_results": [
                {
                    "check_id": check_id,
                    "target_ids": list(check["target_ids"]),
                    "observation_status": "observed",
                    "conclusion": "valid",
                    "reason": "The required relation is visible and valid.",
                }
            ],
        }

    result = _evaluate_shared_packet(
        _group_packet(paths=["global.png", "group-local.png"]),
        call_judge=call_judge,
        audit_records=audit_records,
    )

    assert result["status"] == "evaluated"
    assert len(calls) == 2
    assert calls[0]["render_evidence"] == [
        "global.png",
        "group-local.png",
    ]
    assert calls[1]["render_evidence"] == [
        "global.png",
        "group-local.png",
        "shared-a-b-detail.png",
    ]
    bank = result["functional_group_evidence_bank"]["groups"]["group_001"]
    shared = next(
        item
        for item in bank["artifacts"]
        if item["artifact_id"] == "path:shared-a-b-detail.png"
    )
    assert shared["sources"][0]["source_check_id"] == "check-a-b"
    assert shared["consumer_check_ids"] == ["check-b-c"]
    episodes = result["group_results"][0]["check_episodes"]
    assert [item["functional_check_episode_id"] for item in episodes] == [
        "check-a-b",
        "check-b-c",
    ]


def test_shared_scheduler_fails_closed_when_fixed_seed_is_invalid() -> None:
    calls: list[dict] = []

    def call_judge(_judge, request: dict) -> dict:
        calls.append(request)
        raise AssertionError("Judge must not receive an invalid bank")

    result = _evaluate_shared_packet(
        _group_packet(paths=["global-only.png"]),
        call_judge=call_judge,
        audit_records=[],
    )

    assert calls == []
    assert result["status"] == "failed"
    assert result["terminal_state"] == "infrastructure_failure"
    assert result["judge_call_count"] == 0
    bank = result["functional_group_evidence_bank"]["groups"]["group_001"]
    assert bank["status"] == "unavailable"
    assert bank["reason"] == "group_evidence_bank_validation_failed"
    for episode in result["group_results"][0]["check_episodes"]:
        assert episode["functional_group_evidence_window_audit"][
            "status"
        ] == "failed_closed"
