from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.evaluator.scene_quality.functional_ownership import (
    build_cross_metric_ownership_audit,
    build_functional_ownership_ledger,
)
from benchmark.evaluator.scene_quality.placement_checks import (
    apply_placement_check_judgements,
    build_pending_placement_check,
    build_placement_check_ledger,
    canonicalize_placement_defect_linkage,
    forced_group_ids_from_placement_checks,
    merge_placement_checks,
    normalize_judge_originated_placement_results,
    placement_checks_for_group,
    placement_global_checks,
    validate_placement_check_results,
)


OBJECT_IDS = ["chair", "table", "pendant", "wardrobe"]
GROUPS = [
    {
        "group_id": "dining",
        "object_ids": ["chair", "table", "pendant"],
    },
    {
        "group_id": "storage",
        "object_ids": ["wardrobe"],
    },
]


def _discovery(*candidates: dict) -> dict:
    return {
        "schema_version": "placement_discovery_v2",
        "considered_object_ids": list(OBJECT_IDS),
        "candidates": [deepcopy(item) for item in candidates],
        "reason": "complete",
        "decision_authority": "none",
    }


def _candidate(
    *,
    subject: str,
    check_type: str,
    context: list[str] | None = None,
) -> dict:
    return {
        "subject_id": subject,
        "context_ids": list(context or []),
        "check_type": check_type,
        "observation_goal": f"Inspect {subject} for {check_type}.",
    }


def _valid_row(check: dict) -> dict:
    return {
        "check_id": check["check_id"],
        "subject_id": check["subject_id"],
        "context_ids": deepcopy(check["context_ids"]),
        "observation_status": "observed",
        "conclusion": "valid",
        "reason": "The typed placement obligation is satisfied.",
    }


def _invalid_defect(check: dict) -> dict:
    return {
        "scope": "implausible_local_context",
        "target_ids": [check["subject_id"]],
        "relation": check["check_type"],
        "reason": "The typed placement obligation fails.",
        "severity": "material_contextual_mismatch",
        "check_id": check["check_id"],
        "check_type": check["check_type"],
    }


def test_exact_check_id_canonicalizes_redundant_defect_fields() -> None:
    check = build_placement_check_ledger(
        _discovery(
            _candidate(subject="chair", check_type="scene_zone")
        ),
        groups=GROUPS,
    )["checks"][0]
    result = canonicalize_placement_defect_linkage(
        {
            "defects": [
                {
                    **_invalid_defect(check),
                    "relation": "natural-language location wording",
                    "check_type": "wrong_redundant_value",
                }
            ]
        },
        required_checks=[check],
    )

    assert result["defects"][0]["relation"] == "scene_zone"
    assert result["defects"][0]["check_type"] == "scene_zone"


def test_canonical_candidates_merge_identities_and_route() -> None:
    discovery = _discovery(
        _candidate(
            subject="chair",
            check_type="support_and_height",
        ),
        _candidate(
            subject="chair",
            check_type="support_and_height",
        ),
        _candidate(subject="pendant", check_type="scene_zone"),
        _candidate(
            subject="pendant",
            check_type="contextual_anchor",
            context=["table"],
        ),
        _candidate(
            subject="chair",
            check_type="contextual_anchor",
            context=["wardrobe"],
        ),
    )
    ledger = build_placement_check_ledger(discovery, groups=GROUPS)

    assert ledger["accepted_check_count"] == 4
    support = next(
        item
        for item in ledger["checks"]
        if item["check_type"] == "support_and_height"
    )
    assert support["owner_stage"] == "group_local"
    assert support["owning_group_id"] == "dining"
    assert support["source_check_types"] == ["support_and_height"]
    assert {
        item["subject_id"]
        for item in placement_global_checks(ledger)
    } == {"chair", "pendant"}
    assert {
        item["check_type"]
        for item in placement_checks_for_group(ledger, "dining")
    } == {"support_and_height", "contextual_anchor"}
    assert forced_group_ids_from_placement_checks(ledger) == ["dining"]


def test_legacy_placement_candidate_field_fails_closed() -> None:
    candidate = _candidate(
        subject="chair",
        check_type="support_and_height",
    )
    candidate["observation_kind"] = candidate.pop("check_type")
    with pytest.raises(ValueError, match="unsupported placement check type"):
        build_placement_check_ledger(
            _discovery(candidate),
            groups=GROUPS,
        )


def test_context_ids_are_canonicalized_for_stable_serialization() -> None:
    first = build_placement_check_ledger(
        _discovery(
            _candidate(
                subject="chair",
                check_type="contextual_anchor",
                context=["table", "pendant"],
            )
        ),
        groups=GROUPS,
    )
    second = build_placement_check_ledger(
        _discovery(
            _candidate(
                subject="chair",
                check_type="contextual_anchor",
                context=["pendant", "table"],
            )
        ),
        groups=GROUPS,
    )

    assert first["checks"] == second["checks"]
    assert first["checks"][0]["context_ids"] == ["pendant", "table"]


def test_result_row_context_ids_are_canonicalized_before_persistence() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(
                subject="chair",
                check_type="contextual_anchor",
                context=["pendant", "table"],
            )
        ),
        groups=GROUPS,
    )
    check = ledger["checks"][0]
    row = _valid_row(check)
    row["context_ids"] = ["table", "pendant"]

    resolution = validate_placement_check_results(
        {
            "verdict": "valid",
            "defects": [],
            "placement_check_results": [row],
        },
        required_checks=[check],
    )
    updated, _ = apply_placement_check_judgements(
        ledger,
        global_record=None,
        group_results=[
            {
                "group_id": "dining",
                "placement_check_results": [row],
            }
        ],
    )

    assert resolution["rows"][0]["context_ids"] == ["pendant", "table"]
    assert updated["checks"][0]["result_row"]["context_ids"] == [
        "pendant",
        "table",
    ]


@pytest.mark.parametrize(
    "mutator, error",
    [
        (
            lambda value: value["candidates"][0].update(
                subject_id="unknown"
            ),
            "unknown subject",
        ),
        (
            lambda value: value["candidates"][0].update(
                context_ids=["unknown"]
            ),
            "unknown IDs",
        ),
        (
            lambda value: value["candidates"][0].update(
                context_ids=["chair"]
            ),
            "own context",
        ),
    ],
)
def test_discovery_identity_errors_fail_closed(
    mutator,
    error: str,
) -> None:
    discovery = _discovery(
        _candidate(
            subject="chair",
            check_type="contextual_anchor",
            context=["table"],
        )
    )
    mutator(discovery)

    with pytest.raises(ValueError, match=error):
        build_placement_check_ledger(discovery, groups=GROUPS)


def test_exact_result_rows_and_subject_only_defect_ownership() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(
                subject="pendant",
                check_type="contextual_anchor",
                context=["table"],
            )
        ),
        groups=GROUPS,
    )
    check = ledger["checks"][0]
    row = {
        **_valid_row(check),
        "conclusion": "invalid",
        "reason": "The pendant is not anchored over the table.",
    }
    result = {
        "verdict": "invalid",
        "defects": [_invalid_defect(check)],
        "placement_check_results": [row],
    }

    resolution = validate_placement_check_results(
        result,
        required_checks=[check],
    )

    assert resolution["complete"] is True
    assert resolution["invalid_check_ids"] == [check["check_id"]]

    missing = deepcopy(result)
    missing["placement_check_results"] = []
    with pytest.raises(ValueError, match="exactly once"):
        validate_placement_check_results(
            missing,
            required_checks=[check],
        )

    context_owned = deepcopy(result)
    context_owned["defects"][0]["target_ids"] = ["table"]
    with pytest.raises(ValueError, match="only its subject"):
        validate_placement_check_results(
            context_owned,
            required_checks=[check],
        )

    duplicate = deepcopy(result)
    duplicate["placement_check_results"].append(deepcopy(row))
    with pytest.raises(ValueError, match="exactly once"):
        validate_placement_check_results(
            duplicate,
            required_checks=[check],
        )


def test_function_owned_exclusion_requires_exact_event_reference() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(subject="chair", check_type="scene_zone")
        ),
        groups=GROUPS,
    )
    check = ledger["checks"][0]
    event = {
        "event_id": "functional_event:blocker",
        "affected_object_ids": ["wardrobe"],
        "causal_object_ids": ["chair"],
        "scoring_target_ids": ["chair"],
    }
    row = {
        **_valid_row(check),
        "conclusion": "excluded_function_owned",
        "function_event_ref": event["event_id"],
        "same_physical_event": True,
        "reason": "This is the exact already-scored blocker event.",
    }
    result = {
        "verdict": "valid",
        "defects": [],
        "placement_check_results": [row],
    }

    resolution = validate_placement_check_results(
        result,
        required_checks=[check],
        function_events=[event],
    )

    assert resolution["excluded_function_owned_check_ids"] == [
        check["check_id"]
    ]
    wrong_ref = deepcopy(result)
    wrong_ref["placement_check_results"][0][
        "function_event_ref"
    ] = "functional_event:other"
    with pytest.raises(ValueError, match="unknown functional ownership"):
        validate_placement_check_results(
            wrong_ref,
            required_checks=[check],
            function_events=[event],
        )


def test_placement_invalid_row_keeps_acquisition_open_for_unresolved_check() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(subject="chair", check_type="scene_zone"),
            _candidate(subject="pendant", check_type="scene_zone"),
        ),
        groups=GROUPS,
    )
    invalid_check, unresolved_check = placement_global_checks(ledger)
    invalid_row = {
        **_valid_row(invalid_check),
        "conclusion": "invalid",
        "reason": "The subject occupies an implausible scene zone.",
    }
    unresolved_row = {
        **_valid_row(unresolved_check),
        "observation_status": "missing",
        "conclusion": "unresolved",
        "reason": "The contextual anchor is not jointly visible.",
    }

    resolution = validate_placement_check_results(
        {
            "verdict": "ambiguous",
            "defects": [],
            "placement_check_results": [
                invalid_row,
                unresolved_row,
            ],
        },
        required_checks=[invalid_check, unresolved_check],
    )

    assert resolution["complete"] is False
    assert resolution["invalid_check_ids"] == [
        invalid_check["check_id"]
    ]
    assert resolution["unresolved_check_ids"] == [
        unresolved_check["check_id"]
    ]


def test_judge_originated_check_is_typed_and_phase_scoped() -> None:
    raw = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "semantically_inappropriate_scene_zone",
                "target_ids": ["chair"],
                "relation": "scene_zone",
                "reason": "The chair is in an implausible room zone.",
                "severity": "material_contextual_mismatch",
                "check_id": "proposal-zone",
            }
        ],
        "judge_originated_placement_results": [
            {
                "proposal_id": "proposal-zone",
                "subject_id": "chair",
                "context_ids": [],
                "check_type": "scene_zone",
                "observation_goal": "Inspect the chair's room zone.",
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The chair is in an implausible room zone.",
                "severity": "material_contextual_mismatch",
            }
        ],
    }

    adjusted, checks = normalize_judge_originated_placement_results(
        raw,
        known_ids=set(OBJECT_IDS),
        groups=GROUPS,
        existing_checks=[],
        expected_owner_stage="scene_global",
    )

    assert len(checks) == 1
    assert checks[0]["origin"] == "judge_originated"
    assert adjusted["defects"][0]["target_ids"] == ["chair"]
    assert adjusted["defects"][0]["check_id"] == checks[0]["check_id"]
    assert adjusted["placement_check_results"][0]["check_id"] == (
        checks[0]["check_id"]
    )

    assert "judge_originated_placement_results" not in adjusted
    assert adjusted[
        "judge_originated_placement_check_registrations"
    ] == checks
    assert len(adjusted["placement_check_results"]) == 1

    with pytest.raises(ValueError, match="active 'group_local' phase"):
        normalize_judge_originated_placement_results(
            raw,
            known_ids=set(OBJECT_IDS),
            groups=GROUPS,
            existing_checks=[],
            expected_owner_stage="group_local",
        )


def test_pending_proposal_can_force_singleton_group() -> None:
    check = build_pending_placement_check(
        {
            "proposal_id": "proposal-support",
            "subject_id": "wardrobe",
            "context_ids": [],
            "check_type": "support_and_height",
            "observation_goal": "Inspect support and placement height.",
        },
        known_ids=set(OBJECT_IDS),
        groups=GROUPS,
        source_ref="judge-call-1",
    )
    ledger = merge_placement_checks(
        {
            "schema_version": "placement_check_ledger_v1",
            "checks": [],
            "accepted_check_count": 0,
            "decision_authority": "none",
        },
        [check],
    )

    assert check["owner_stage"] == "group_local"
    assert check["owning_group_id"] == "storage"
    assert forced_group_ids_from_placement_checks(ledger) == ["storage"]


def test_clearance_blocker_projects_one_functional_ownership_event() -> None:
    functional_ledger = {
        "schema_version": "functional_check_ledger_v1",
        "checks": [
            {
                "check_id": "clearance:wardrobe",
                "check_type": "clearance",
                "target_ids": ["wardrobe"],
                "check_conclusion": "invalid",
                "judge_result_ref": "group_local_review:storage",
                "scoring_target_ids": ["chair"],
                "result_row": {
                    "check_id": "clearance:wardrobe",
                    "target_ids": ["wardrobe"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The chair blocks the wardrobe frontage.",
                    "affected_object_ids": ["wardrobe"],
                    "cause_kind": "external_object",
                    "causal_object_ids": ["chair"],
                    "scoring_target_ids": ["chair"],
                },
            }
        ],
    }
    group_result = {
        "group_id": "storage",
        "score": 0.0,
        "judgement": {
            "verdict": "invalid",
            "confidence": 0.93,
            "reason": "The chair blocks the wardrobe frontage.",
            "defects": [
                {
                    "scope": "clearance",
                    "target_ids": ["chair"],
                    "relation": "clearance",
                    "reason": "The chair blocks ordinary wardrobe use.",
                }
            ],
            "provenance": {"decision_ref": "judge:storage:clearance"},
        },
    }

    ownership = build_functional_ownership_ledger(
        scene_object_ids=OBJECT_IDS,
        global_record=None,
        relation_results=[],
        group_results=[group_result],
        functional_check_ledger=functional_ledger,
    )

    assert ownership["event_count"] == 1
    event = ownership["events"][0]
    assert event["affected_object_ids"] == ["wardrobe"]
    assert event["causal_object_ids"] == ["chair"]
    assert event["scoring_target_ids"] == ["chair"]
    assert event["cause_kind"] == "external_object"


def test_pair_check_reference_survives_object_level_defect_attribution() -> None:
    functional_ledger = {
        "schema_version": "functional_check_ledger_v5",
        "checks": [
            {
                "check_id": "direction:chair-table",
                "check_type": "directional_correspondence",
                "target_ids": ["chair", "table"],
                "check_conclusion": "invalid",
                "judge_result_ref": "group_local_review:dining",
                "result_row": {
                    "check_id": "direction:chair-table",
                    "target_ids": ["chair", "table"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The chair faces away from the table.",
                },
            }
        ],
    }
    group_result = {
        "group_id": "dining",
        "score": 0.0,
        "judgement": {
            "verdict": "invalid",
            "confidence": 0.9,
            "reason": "The chair cannot support ordinary table use.",
            "defects": [
                {
                    "scope": "functional_relation",
                    "target_ids": ["chair"],
                    "relation": "directional_correspondence",
                    "reason": "The chair faces away from the table.",
                    "check_refs": ["direction:chair-table"],
                }
            ],
        },
    }

    ownership = build_functional_ownership_ledger(
        scene_object_ids=OBJECT_IDS,
        global_record=None,
        relation_results=[],
        group_results=[group_result],
        functional_check_ledger=functional_ledger,
    )

    assert ownership["event_count"] == 1
    event = ownership["events"][0]
    assert event["check_refs"] == ["direction:chair-table"]
    assert event["scoring_target_ids"] == ["chair"]


def test_per_check_episode_reference_projects_to_group_ownership_phase() -> None:
    check_id = "direction:chair-table"
    functional_ledger = {
        "schema_version": "functional_check_ledger_v5",
        "checks": [
            {
                "check_id": check_id,
                "check_type": "directional_correspondence",
                "target_ids": ["chair", "table"],
                "check_conclusion": "invalid",
                "judge_result_ref": (
                    "group_local_review:dining:check:"
                    f"{check_id}"
                ),
                "result_row": {
                    "check_id": check_id,
                    "target_ids": ["chair", "table"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The chair faces away from the table.",
                },
            }
        ],
    }
    group_result = {
        "group_id": "dining",
        "score": 0.0,
        "judgement": {
            "verdict": "invalid",
            "confidence": 0.9,
            "reason": "The chair cannot support ordinary table use.",
            "defects": [
                {
                    "scope": "functional_relation",
                    "target_ids": ["chair"],
                    "relation": "directional_correspondence",
                    "reason": "The chair faces away from the table.",
                    "check_refs": [check_id],
                }
            ],
        },
    }

    ownership = build_functional_ownership_ledger(
        scene_object_ids=OBJECT_IDS,
        global_record=None,
        relation_results=[],
        group_results=[group_result],
        functional_check_ledger=functional_ledger,
    )

    assert ownership["event_count"] == 1
    assert ownership["events"][0]["check_refs"] == [check_id]
    assert ownership["events"][0]["source_phase"] == (
        "group_local_review:dining"
    )


def test_placement_lifecycle_and_cross_metric_audit_preserve_independent_defect() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(subject="chair", check_type="scene_zone"),
            _candidate(
                subject="pendant",
                check_type="contextual_anchor",
                context=["table"],
            ),
        ),
        groups=GROUPS,
    )
    chair_check = next(
        item for item in ledger["checks"] if item["subject_id"] == "chair"
    )
    pendant_check = next(
        item
        for item in ledger["checks"]
        if item["subject_id"] == "pendant"
    )
    function_event = {
        "event_id": "functional_event:blocker",
        "affected_object_ids": ["wardrobe"],
        "causal_object_ids": ["chair"],
        "scoring_target_ids": ["chair"],
    }
    global_record = {
        "placement_check_results": [
            {
                **_valid_row(chair_check),
                "conclusion": "excluded_function_owned",
                "function_event_ref": function_event["event_id"],
                "same_physical_event": True,
                "reason": "Exact blocker event already belongs to Function.",
            },
            {
                **_valid_row(pendant_check),
                "conclusion": "invalid",
                "reason": "The pendant is not anchored over the table.",
            },
        ]
    }
    updated, coverage = apply_placement_check_judgements(
        ledger,
        global_record=global_record,
        group_results=[],
    )
    audit = build_cross_metric_ownership_audit(
        functional_ownership_ledger={
            "events": [function_event],
        },
        placement_check_ledger=updated,
    )

    assert coverage["complete"] is True
    assert coverage["excluded_function_owned_check_ids"] == [
        chair_check["check_id"]
    ]
    assert coverage["invalid_check_ids"] == [pendant_check["check_id"]]
    assert audit["excluded_placement_checks"] == [
        {
            "placement_check_id": chair_check["check_id"],
            "function_event_ref": function_event["event_id"],
            "subject_id": "chair",
            "same_physical_event": True,
            "decision_authority": "none",
        }
    ]
    assert audit["independent_invalid_placement_check_ids"] == [
        pendant_check["check_id"]
    ]


def test_normal_scene_resolves_every_typed_placement_check_valid() -> None:
    ledger = build_placement_check_ledger(
        _discovery(
            _candidate(subject="chair", check_type="scene_zone"),
            _candidate(subject="chair", check_type="support_and_height"),
            _candidate(
                subject="chair",
                check_type="contextual_anchor",
                context=["table"],
            ),
        ),
        groups=GROUPS,
    )
    global_checks = placement_global_checks(ledger)
    local_checks = placement_checks_for_group(ledger, "dining")

    updated, coverage = apply_placement_check_judgements(
        ledger,
        global_record={
            "placement_check_results": [
                _valid_row(check) for check in global_checks
            ]
        },
        group_results=[
            {
                "group_id": "dining",
                "placement_check_results": [
                    _valid_row(check) for check in local_checks
                ],
            }
        ],
    )

    assert coverage["required_check_count"] == 3
    assert coverage["resolved_check_count"] == 3
    assert coverage["complete"] is True
    assert coverage["invalid_check_ids"] == []
    assert coverage["excluded_function_owned_check_ids"] == []
    assert all(
        check["lifecycle_status"] == "resolved"
        and check["check_conclusion"] == "valid"
        and check["result_row"]["reason"]
        for check in updated["checks"]
    )
