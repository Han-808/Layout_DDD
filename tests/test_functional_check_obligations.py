from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.evaluator.scene_quality.functional_acquisition import (
    build_functional_acquisition_plan,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    apply_functional_check_judgements,
    build_functional_check_ledger,
    canonicalize_clearance_causal_attribution,
    canonicalize_functional_defect_check_linkage,
    canonicalize_typed_invalid_envelope,
    checks_for_group,
    forced_group_ids_from_checks,
    salvage_functional_judge_response,
    update_functional_check_evidence,
    validate_functional_check_results,
)
from benchmark.evaluator.scene_quality.cross_group_relations import (
    _cross_group_relation_episode_specs,
    _evaluate_cross_group_relation_scopes,
    reconcile_directional_relation_conflicts,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    functional_relation_judge_packet,
)
from benchmark.evaluator.scene_quality.global_group_first import (
    _bind_architecture_orientation_evidence,
)
from benchmark.evaluator.scene_quality.group_scoped import (
    _expand_functional_check_packets,
    _functional_visual_preflight,
    _scope_functional_episode_evidence,
)
from benchmark.visual_judge.functional_evidence import (
    FUNCTIONAL_PROBE_MAX_UNITS,
)


GROUPS = [
    {
        "group_id": "dining",
        "object_ids": ["chair_1", "chair_2", "table"],
    },
    {"group_id": "storage", "object_ids": ["bookshelf"]},
    {"group_id": "media", "object_ids": ["television"]},
    {"group_id": "seating", "object_ids": ["sofa"]},
    {"group_id": "lighting", "object_ids": ["lamp"]},
]


def test_cross_group_relation_without_any_visual_returns_defaulted_binary(
) -> None:
    result = _evaluate_cross_group_relation_scopes(
        specs=[
            {
                "relation_id": "sofa_tv",
                "target_ids": ["sofa", "television"],
                "group_ids": ["seating", "media"],
                "pair_specific_evidence_available": False,
                "retained_evidence_forced_choice_available": False,
                "required_checks": [
                    {
                        "check_id": "functional_check_001",
                        "check_type": "directional_correspondence",
                        "target_ids": ["sofa", "television"],
                        "required_observations": [
                            "joint_visibility",
                            "front_back_disambiguated",
                        ],
                    }
                ],
                "required_check_ids": ["functional_check_001"],
                "acquisition_status": "failed",
                "acquisition_error": {
                    "error_type": "RuntimeError",
                    "error": "no feasible visual candidate",
                },
            }
        ],
        metric_name="functional_consistency",
        scene={"objects": []},
        global_evidence=[],
        vlm_judge=None,
        prompt=None,
        visual_style_spec=None,
        authorized_deviations=[],
        build_judge_request=lambda **kwargs: pytest.fail(
            "Judge request must not be built without any visual"
        ),
        call_judge=lambda *args, **kwargs: pytest.fail(
            "Judge must not be invoked without any visual"
        ),
        apply_prompt_exemptions=lambda *args, **kwargs: pytest.fail(
            "no Judge response exists"
        ),
        normalize_judgement=lambda *args, **kwargs: pytest.fail(
            "no Judge response exists"
        ),
    )[0]

    assert result["status"] == "evaluated"
    assert result["score"] == 1.0
    assert result["terminal_state"] == "evaluated_degraded"
    assert result["vlm_invoked"] is False
    assert result["judgement"]["defaulted"] is True
    assert result["judgement"]["functional_check_results"] == [
        {
            "check_id": "functional_check_001",
            "target_ids": ["sofa", "television"],
            "observation_status": "missing",
            "conclusion": "valid",
            "reason": (
                "No visual artifact was available for this required check; "
                "the terminal policy returned a zero-confidence default "
                "without claiming evidence coverage."
            ),
        }
    ]
    assert result["evidence_coverage"]["grounded"] is False


def test_batched_functional_salvage_keeps_initial_invalid_and_repairs_bad_row(
) -> None:
    checks = [
        {
            "check_id": "functional_check_001",
            "check_type": "architecture_orientation",
            "target_ids": ["bookshelf"],
        },
        {
            "check_id": "functional_check_002",
            "check_type": "architecture_orientation",
            "target_ids": ["television"],
        },
    ]
    defect = {
        "scope": "functional_access",
        "target_ids": ["bookshelf"],
        "relation": "usable_side_orientation",
        "reason": "The ordinary access side faces the boundary.",
        "check_refs": ["functional_check_001"],
    }
    initial = {
        "confidence": 0.8,
        "defects": [defect],
        "functional_check_results": [
            {
                "check_id": "functional_check_001",
                "target_ids": ["bookshelf"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The ordinary access side faces the boundary.",
            },
            {
                "check_id": "functional_check_002",
                "target_ids": ["television"],
                "observation_status": "observed",
                "conclusion": "not_a_conclusion",
                "reason": "Malformed transport token.",
            },
        ],
    }
    repair = {
        "confidence": 0.6,
        "defects": [],
        "functional_check_results": [
            {
                "check_id": "functional_check_001",
                "target_ids": ["bookshelf"],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "Repair attempted to revise a legal atom.",
            },
            {
                "check_id": "functional_check_002",
                "target_ids": ["television"],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "The second check was repaired.",
            },
        ],
    }

    result = salvage_functional_judge_response(
        repair,
        required_checks=checks,
        fallback_value=initial,
    )

    rows = {
        row["check_id"]: row
        for row in result["functional_check_results"]
    }
    assert rows["functional_check_001"]["conclusion"] == "invalid"
    assert rows["functional_check_002"]["conclusion"] == "valid"
    assert result["functional_salvage_audit"]["accepted_sources"] == {
        "functional_check_001": "initial",
        "functional_check_002": "repair",
    }
    assert result["verdict"] == "invalid"


def test_multi_check_union_defect_is_split_into_atomic_linkages() -> None:
    checks = [
        {
            "check_id": "functional_check_001",
            "check_type": "within_group_correspondence",
            "target_ids": ["armchair_01", "coffee_table"],
        },
        {
            "check_id": "functional_check_002",
            "check_type": "within_group_correspondence",
            "target_ids": ["armchair_02", "coffee_table"],
        },
    ]
    result = {
        "verdict": "invalid",
        "functional_check_results": [
            {
                "check_id": check["check_id"],
                "target_ids": check["target_ids"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The chair does not support the shared task.",
            }
            for check in checks
        ],
        "defects": [
            {
                "scope": "group_local",
                "target_ids": [
                    "armchair_01",
                    "armchair_02",
                    "coffee_table",
                ],
                "relation": "functional_correspondence",
                "reason": "Both chair-table checks fail.",
                "check_refs": [
                    "functional_check_001",
                    "functional_check_002",
                ],
            }
        ],
    }

    normalized = canonicalize_functional_defect_check_linkage(
        result,
        required_checks=checks,
    )

    assert [item["check_refs"] for item in normalized["defects"]] == [
        ["functional_check_001"],
        ["functional_check_002"],
    ]
    assert [item["target_ids"] for item in normalized["defects"]] == [
        ["armchair_01", "coffee_table"],
        ["armchair_02", "coffee_table"],
    ]
    assert validate_functional_check_results(
        normalized,
        required_checks=checks,
    )["complete"] is True


def test_clearance_attribution_is_copied_from_linked_defect_before_repair() -> None:
    checks = [
        {
            "check_id": "functional_check_010",
            "check_type": "architecture_orientation",
            "target_ids": ["tv_stand"],
        },
        {
            "check_id": "functional_check_015",
            "check_type": "clearance",
            "target_ids": ["tv_stand"],
            "allowed_causal_object_ids": ["television", "tv_stand"],
        },
    ]
    result = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "opening_clearance",
                "target_ids": ["tv_stand"],
                "relation": "frontage_toward_boundary",
                "reason": "The authored frontage is not usable.",
                "check_refs": [
                    "functional_check_010",
                    "functional_check_015",
                ],
                "affected_object_ids": ["tv_stand"],
                "cause_kind": "self_layout",
                "causal_object_ids": ["tv_stand"],
                "scoring_target_ids": ["tv_stand"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_010",
                "target_ids": ["tv_stand"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The frontage points toward the boundary.",
            },
            {
                "check_id": "functional_check_015",
                "target_ids": ["tv_stand"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The approach region is unavailable.",
            },
        ],
    }

    normalized = canonicalize_clearance_causal_attribution(
        result,
        required_checks=checks,
    )

    clearance_row = normalized["functional_check_results"][1]
    assert clearance_row["affected_object_ids"] == ["tv_stand"]
    assert clearance_row["cause_kind"] == "self_layout"
    assert clearance_row["causal_object_ids"] == ["tv_stand"]
    assert clearance_row["scoring_target_ids"] == ["tv_stand"]
    assert validate_functional_check_results(
        normalized,
        required_checks=checks,
    )["complete"] is True


def test_empty_self_layout_causal_ids_are_deterministically_completed() -> None:
    checks = [
        {
            "check_id": "functional_check_005",
            "check_type": "architecture_orientation",
            "target_ids": ["toilet"],
        },
        {
            "check_id": "functional_check_011",
            "check_type": "clearance",
            "target_ids": ["toilet"],
            "allowed_causal_object_ids": ["toilet", "trash_bin"],
        },
    ]
    result = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["toilet"],
                "relation": "usable_side_clearance",
                "reason": "The seating approach region is unavailable.",
                "check_refs": [
                    "functional_check_005",
                    "functional_check_011",
                ],
                "affected_object_ids": ["toilet"],
                "cause_kind": "self_layout",
                "causal_object_ids": [],
                "scoring_target_ids": ["toilet"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_005",
                "target_ids": ["toilet"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The usable side points outside the room.",
            },
            {
                "check_id": "functional_check_011",
                "target_ids": ["toilet"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "There is no usable approach region.",
                "affected_object_ids": ["toilet"],
                "cause_kind": "self_layout",
                "causal_object_ids": [],
                "scoring_target_ids": ["toilet"],
            },
        ],
    }

    normalized = canonicalize_clearance_causal_attribution(
        result,
        required_checks=checks,
    )

    assert normalized["defects"][0]["causal_object_ids"] == ["toilet"]
    assert normalized["functional_check_results"][1][
        "causal_object_ids"
    ] == ["toilet"]
    assert validate_functional_check_results(
        normalized,
        required_checks=checks,
    )["complete"] is True


@pytest.mark.parametrize(
    ("affected_id", "blocker_id", "check_id"),
    [
        ("dartboard", "pool_table", "functional_check_013"),
        ("pool_table", "sofa", "functional_check_016"),
    ],
)
def test_external_blocker_drives_derived_scoring_attribution(
    affected_id: str,
    blocker_id: str,
    check_id: str,
) -> None:
    """Recorded N025 shapes retain the semantic invalid conclusion."""

    checks = [
        {
            "check_id": check_id,
            "check_type": "clearance",
            "target_ids": [affected_id],
            "allowed_causal_object_ids": [affected_id, blocker_id],
        }
    ]
    result = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                # The model used the affected object as the defect target and
                # omitted redundant attribution fields, as in the real run.
                "target_ids": [affected_id],
                "relation": "usable_side_clearance",
                "reason": f"{blocker_id} blocks ordinary use.",
                "check_refs": [check_id],
            }
        ],
        "functional_check_results": [
            {
                "check_id": check_id,
                "target_ids": [affected_id],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": f"{blocker_id} occupies the required use zone.",
                "affected_object_ids": [affected_id],
                "cause_kind": "external_object",
                "causal_object_ids": [blocker_id],
                # This is derived bookkeeping, not a second semantic claim.
                "scoring_target_ids": [affected_id],
            }
        ],
    }

    normalized = canonicalize_clearance_causal_attribution(
        result,
        required_checks=checks,
    )

    row = normalized["functional_check_results"][0]
    defect = normalized["defects"][0]
    assert row["causal_object_ids"] == [blocker_id]
    assert row["scoring_target_ids"] == [blocker_id]
    assert defect["causal_object_ids"] == [blocker_id]
    assert defect["scoring_target_ids"] == [blocker_id]
    assert defect["target_ids"] == [blocker_id]
    assert validate_functional_check_results(
        normalized,
        required_checks=checks,
    )["complete"] is True


def test_external_blocker_normalization_fails_closed_on_conflicting_identity() -> None:
    checks = [
        {
            "check_id": "functional_check_013",
            "check_type": "clearance",
            "target_ids": ["dartboard"],
            "allowed_causal_object_ids": [
                "dartboard",
                "pool_table",
                "sofa",
            ],
        }
    ]
    result = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "opening_clearance",
                "target_ids": ["pool_table"],
                "relation": "usable_side_clearance",
                "reason": "A blocker occupies the required use zone.",
                "check_refs": ["functional_check_013"],
                "affected_object_ids": ["dartboard"],
                "cause_kind": "external_object",
                "causal_object_ids": ["sofa"],
                "scoring_target_ids": ["pool_table"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_013",
                "target_ids": ["dartboard"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The pool table occupies the required use zone.",
                "affected_object_ids": ["dartboard"],
                "cause_kind": "external_object",
                "causal_object_ids": ["pool_table"],
                "scoring_target_ids": ["pool_table"],
            }
        ],
    }

    normalized = canonicalize_clearance_causal_attribution(
        result,
        required_checks=checks,
    )
    assert normalized == result
    with pytest.raises(ValueError, match="conflicting causal_object_ids"):
        validate_functional_check_results(
            normalized,
            required_checks=checks,
        )


def test_external_blocker_normalization_fails_closed_on_unknown_identity() -> None:
    checks = [
        {
            "check_id": "functional_check_013",
            "check_type": "clearance",
            "target_ids": ["dartboard"],
            "allowed_causal_object_ids": ["dartboard", "pool_table"],
        }
    ]
    result = {
        "verdict": "invalid",
        "defects": [
            {
                "scope": "opening_clearance",
                "target_ids": ["ghost"],
                "relation": "usable_side_clearance",
                "reason": "An unknown object is claimed as the blocker.",
                "check_refs": ["functional_check_013"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_013",
                "target_ids": ["dartboard"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "An unknown object is claimed as the blocker.",
                "affected_object_ids": ["dartboard"],
                "cause_kind": "external_object",
                "causal_object_ids": ["ghost"],
                "scoring_target_ids": ["ghost"],
            }
        ],
    }

    normalized = canonicalize_clearance_causal_attribution(
        result,
        required_checks=checks,
    )
    assert normalized == result
    with pytest.raises(ValueError, match="unknown causal objects"):
        validate_functional_check_results(
            normalized,
            required_checks=checks,
        )


def test_cross_group_episode_rejects_non_atomic_legacy_relation() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        _cross_group_relation_episode_specs(
            acquisition_audit={
                "planner_mode": "functional_discovery",
                "functional_discovery": {
                    "cross_group_correspondences": [
                        {
                            "discovery_id": "legacy_cluster",
                            "target_ids": [
                                "sofa",
                                "television",
                                "lamp",
                            ],
                            "predicate": "directional_correspondence",
                            "observation_goal": "Observe the relation.",
                        }
                    ]
                },
                "probe_results": [],
            },
            groups=GROUPS,
            global_paths=["/evidence/global.png"],
        )


def test_final_typed_defect_canonicalizes_metric_envelope() -> None:
    original = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "missing_evidence": ["target_visible"],
        "evidence_request": {
            "target_ids": ["bookshelf"],
            "missing_observations": ["target_visible"],
            "view_goal": "show the target",
            "metadata": {},
        },
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["bookshelf"],
                "relation": "architecture_orientation",
                "reason": "The usable side faces inaccessible space.",
                "check_refs": ["functional_check_001"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_001",
                "target_ids": ["bookshelf"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The usable side faces inaccessible space.",
            }
        ],
    }

    normalized = canonicalize_typed_invalid_envelope(original)

    assert normalized["verdict"] == "invalid"
    assert normalized["evidence_status"] == "sufficient"
    assert normalized["missing_evidence"] == []
    assert normalized["evidence_request"] is None
    assert original["verdict"] == "ambiguous"


def test_untyped_defect_cannot_canonicalize_insufficient_envelope() -> None:
    original = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "missing_evidence": ["target_visible"],
        "evidence_request": None,
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["bookshelf"],
                "relation": "architecture_orientation",
                "reason": "Premature untyped claim.",
            }
        ],
        "functional_check_results": [],
    }

    assert canonicalize_typed_invalid_envelope(original) is original


def test_typed_invalid_row_does_not_finalize_while_another_row_is_unresolved() -> None:
    original = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "missing_evidence": ["joint_visibility"],
        "evidence_request": {
            "target_ids": ["bookshelf"],
            "missing_observations": ["joint_visibility"],
            "view_goal": "show the remaining relation",
            "metadata": {},
        },
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["bookshelf"],
                "relation": "architecture_orientation",
                "reason": "The observed side faces inaccessible space.",
                "check_refs": ["functional_check_001"],
            }
        ],
        "functional_check_results": [
            {
                "check_id": "functional_check_001",
                "target_ids": ["bookshelf"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The observed side faces inaccessible space.",
            },
            {
                "check_id": "functional_check_002",
                "target_ids": ["bookshelf"],
                "observation_status": "missing",
                "conclusion": "unresolved",
                "reason": "The approach region is hidden.",
            },
        ],
    }

    assert canonicalize_typed_invalid_envelope(original) is original


def _base_discovery() -> dict:
    return {
        "schema_version": "functional_discovery_v3",
        "inspected_object_ids": [
            "chair_1",
            "chair_2",
            "table",
            "bookshelf",
            "television",
            "sofa",
            "lamp",
        ],
        "directed_surface_targets": [],
        "within_group_correspondences": [],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }


def _cross_relation(
    discovery_id: str,
    left: str,
    right: str,
    left_group: str,
    right_group: str,
) -> dict:
    return {
        "discovery_id": discovery_id,
        "target_ids": [left, right],
        "group_ids": [left_group, right_group],
        "observation_kinds": ["mutual_orientation"],
        "observation_goal": "show both interaction sides and relative facing",
    }


def test_n002_within_group_relation_cannot_disappear_after_discovery() -> None:
    discovery = _base_discovery()
    discovery["within_group_correspondences"] = [
        {
            "discovery_id": "chair_table_correspondence",
            "target_ids": ["chair_1", "chair_2", "table"],
            "group_ids": ["dining"],
            "observation_kinds": [
                "mutual_orientation",
                "shared_task_reach",
            ],
            "observation_goal": (
                "show chair interaction sides relative to the table"
            ),
        }
    ]
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        )
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=1,
        groups=GROUPS,
    )
    dining_checks = checks_for_group(
        plan["functional_check_ledger"],
        "dining",
    )

    assert [item["check_type"] for item in dining_checks] == [
        "directional_correspondence",
        "relative_use_geometry",
    ]
    assert all(
        item["source_discovery_ids"] == ["chair_table_correspondence"]
        for item in dining_checks
    )
    assert all(
        item["acquisition_status"] == "judge_requested_on_demand"
        for item in dining_checks
    )
    assert all(
        item["requires_dedicated_acquisition"] is False
        for item in dining_checks
    )
    assert plan["lazy_group_acquisition"]["required_check_ids"] == [
        item["check_id"] for item in dining_checks
    ]
    cross_checks = [
        item
        for item in plan["functional_check_ledger"]["checks"]
        if item["owner_stage"] == "cross_group_relation"
    ]
    assert cross_checks
    assert all(
        item["acquisition_status"] == "scheduled"
        for item in cross_checks
    )

    with pytest.raises(ValueError, match="functional_check_results"):
        validate_functional_check_results(
            {
                "evidence_status": "sufficient",
                "verdict": "valid",
                "defects": [],
            },
            required_checks=dining_checks,
        )


def test_relation_predicates_share_target_set_unit_without_conflating_views() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "tv_surface",
            "target_id": "television",
            "owning_group_id": "media",
            "surface_roles": ["display_side"],
            "need_clearance": False,
            "observation_goal": "show the display side",
        },
        {
            "discovery_id": "sofa_surface",
            "target_id": "sofa",
            "owning_group_id": "seating",
            "surface_roles": ["seating_side"],
            "need_clearance": False,
            "observation_goal": "show the seating side",
        },
    ]
    discovery["cross_group_correspondences"] = [
        {
            "discovery_id": "tv_sofa_geometry",
            "target_ids": ["television", "sofa"],
            "group_ids": ["media", "seating"],
            "predicate": "relative_use_geometry",
            "observation_goal": "show relative joint-use geometry",
        }
    ]

    relative_plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
        groups=GROUPS,
    )
    relative_unit = next(
        item
        for item in relative_plan["probe_units"]
        if item["route_scope"] == "cross_group"
    )
    assert relative_unit["relation_predicates"] == [
        "relative_use_geometry"
    ]
    assert relative_unit["surface_targets"] == []
    assert relative_unit["scheduling_coverage_gain"][
        "new_directed_object_ids"
    ] == []
    assert "relative_layout_visible" in relative_unit[
        "required_observations"
    ]
    assert "front_back_disambiguated" not in relative_unit[
        "required_observations"
    ]

    discovery["cross_group_correspondences"].append(
        {
            "discovery_id": "tv_sofa_direction",
            "target_ids": ["sofa", "television"],
            "group_ids": ["seating", "media"],
            "predicate": "directional_correspondence",
            "observation_goal": "show compatible functional-side directions",
        }
    )
    combined_plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
        groups=GROUPS,
    )
    cross_units = [
        item
        for item in combined_plan["probe_units"]
        if item["route_scope"] == "cross_group"
    ]

    assert len(cross_units) == 1
    assert set(cross_units[0]["relation_predicates"]) == {
        "directional_correspondence",
        "relative_use_geometry",
    }
    assert {
        item["target_id"] for item in cross_units[0]["surface_targets"]
    } == {"television", "sofa"}
    assert set(
        cross_units[0]["scheduling_coverage_gain"][
            "new_directed_object_ids"
        ]
    ) == {"television", "sofa"}
    assert len(cross_units[0]["check_ids"]) == 2


def test_n011_scheduler_does_not_starve_accepted_cross_group_relations() -> None:
    discovery = _base_discovery()
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            f"cross_{index}",
            "television",
            target,
            "media",
            group_id,
        )
        for index, (target, group_id) in enumerate(
            [
                ("sofa", "seating"),
                ("chair_1", "dining"),
                ("chair_2", "dining"),
                ("lamp", "lighting"),
            ],
            start=1,
        )
    ]
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": True,
            "observation_goal": "show the accessible shelf frontage",
        }
    ]
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "bookshelf_approach",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "need_clearance": True,
            "observation_goal": (
                "show shelf frontage and the adjacent approach region"
            ),
        }
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
        groups=GROUPS,
    )

    assert len(plan["probe_units"]) == 4
    assert all(
        item.get("route_scope") == "cross_group"
        for item in plan["probe_units"]
    )
    assert plan["budget"]["cross_group_reservation_complete"] is True
    assert any(
        "bookshelf_approach" in item.get("discovery_ids", [])
        for item in plan["backfill_probe_units"]
    )
    storage_checks = checks_for_group(
        plan["functional_check_ledger"],
        "storage",
    )
    assert {item["check_type"] for item in storage_checks} == {
        "architecture_orientation",
        "clearance",
    }
    orientation_check = next(
        item
        for item in storage_checks
        if item["check_type"] == "architecture_orientation"
    )
    clearance_check = next(
        item
        for item in storage_checks
        if item["check_type"] == "clearance"
    )
    assert orientation_check["source_discovery_ids"] == [
        "bookshelf_surface"
    ]
    assert clearance_check["source_discovery_ids"] == [
        "bookshelf_approach",
        "bookshelf_surface",
    ]
    assert orientation_check["surface_roles"] == ["access_side"]
    assert clearance_check["surface_roles"] == ["access_side"]
    storage_unit = next(
        item
        for item in plan["backfill_probe_units"]
        if item.get("owning_group_id") == "storage"
    )
    assert set(storage_unit["check_ids"]) == {
        orientation_check["check_id"],
        clearance_check["check_id"],
    }
    assert storage_unit["evidence_reuse"][
        "reuses_side_conditioned_view_for_architecture_orientation"
    ] is True
    assert storage_unit["evidence_reuse"][
        "reuses_side_conditioned_view_for_clearance"
    ] is True
    assert "storage" in forced_group_ids_from_checks(
        plan["functional_check_ledger"]
    )


def test_scheduler_reserves_every_accepted_cross_group_relation() -> None:
    discovery = _base_discovery()
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        ),
        _cross_relation(
            "tv_chair_1",
            "television",
            "chair_1",
            "media",
            "dining",
        ),
        _cross_relation(
            "tv_chair_2",
            "television",
            "chair_2",
            "media",
            "dining",
        ),
    ]
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": False,
            "observation_goal": "show the shelf frontage",
        },
        {
            "discovery_id": "lamp_surface",
            "target_id": "lamp",
            "owning_group_id": "lighting",
            "surface_roles": ["control_side"],
            "need_clearance": False,
            "observation_goal": "show the lamp control side",
        },
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=3,
        groups=GROUPS,
    )

    assert sum(
        unit["route_scope"] == "cross_group"
        for unit in plan["probe_units"]
    ) == 3
    selected_targets = {
        target_id
        for unit in plan["probe_units"]
        for target_id in [
            *unit.get("target_ids", []),
            *unit.get("related_target_ids", []),
        ]
    }
    assert {"bookshelf", "lamp"}.isdisjoint(selected_targets)
    assert plan["budget"]["cross_group_reservation_complete"] is True
    assert all(
        unit["scheduling_coverage_gain"].get("reservation")
        == "accepted_cross_group_relation"
        for unit in plan["probe_units"]
    )


def test_functional_probe_budget_is_hard_capped_at_thirty_two() -> None:
    object_ids = [f"object_{index:02d}" for index in range(40)]
    groups = [
        {"group_id": f"group_{index:02d}", "object_ids": [object_id]}
        for index, object_id in enumerate(object_ids)
    ]
    discovery = _base_discovery()
    discovery["inspected_object_ids"] = object_ids
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": f"surface_{object_id}",
            "target_id": object_id,
            "owning_group_id": next(
                group["group_id"]
                for group in groups
                if object_id in group["object_ids"]
            ),
            "surface_roles": ["interaction_side"],
            "need_clearance": False,
            "observation_goal": f"show {object_id} interaction side",
        }
        for object_id in object_ids
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=99,
        groups=groups,
    )

    assert plan["max_probe_units"] == FUNCTIONAL_PROBE_MAX_UNITS == 32
    assert len(plan["probe_units"]) == 32
    assert plan["budget"]["max_probe_units"] == {
        "requested": 99,
        "effective": 32,
        "hard_cap": 32,
        "clamped_to_hard_cap": True,
        "source": "caller",
    }


def test_directed_side_and_direction_independent_clearance_both_route() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": False,
            "observation_goal": "show the accessible shelf frontage",
        }
    ]
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "table_clearance",
            "target_id": "table",
            "owning_group_id": "dining",
            "need_clearance": True,
            "observation_goal": "show the ordinary approach region",
        }
    ]

    ledger = build_functional_check_ledger(discovery, groups=GROUPS)
    storage_checks = checks_for_group(ledger, "storage")
    dining_checks = checks_for_group(ledger, "dining")

    assert [item["check_type"] for item in storage_checks] == [
        "architecture_orientation"
    ]
    assert storage_checks[0]["need_clearance"] is False
    assert "approach_zone_visible" not in storage_checks[0][
        "required_observations"
    ]
    assert "global_context_preserved" in storage_checks[0][
        "required_observations"
    ]
    assert "architecture_plane_visible" in storage_checks[0][
        "required_observations"
    ]
    assert [item["check_type"] for item in dining_checks] == [
        "clearance"
    ]
    assert dining_checks[0]["need_clearance"] is True
    assert "approach_zone_visible" in dining_checks[0][
        "required_observations"
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
        groups=GROUPS,
    )
    directed_unit = next(
        item
        for item in plan["probe_units"]
        if item.get("target_ids") == ["bookshelf"]
    )
    clearance_unit = next(
        item
        for item in plan["probe_units"]
        if item.get("target_ids") == ["table"]
    )
    assert directed_unit["evidence_reuse"]["side_conditioned"] is True
    assert directed_unit["evidence_reuse"][
        "reuses_side_conditioned_view_for_clearance"
    ] is False
    assert clearance_unit["evidence_reuse"]["side_conditioned"] is False
    assert clearance_unit["evidence_reuse"][
        "clearance_observation_frame"
    ] == "object_surrounding_region"


def test_directed_clearance_declaration_requires_downstream_target() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": True,
            "observation_goal": "show the accessible shelf frontage",
        }
    ]

    with pytest.raises(ValueError, match="must match"):
        build_functional_check_ledger(discovery, groups=GROUPS)


def test_every_accepted_discovery_source_reaches_a_typed_judge_route() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "tv_surface",
            "target_id": "television",
            "owning_group_id": "media",
            "surface_roles": ["display_side"],
            "need_clearance": False,
            "observation_goal": "show the display-facing side",
        }
    ]
    discovery["boundary_sensitive_targets"] = [
        {
            "discovery_id": "shelf_boundary",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "observation_goal": "show shelf access relative to room architecture",
        }
    ]
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "shelf_clearance",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "need_clearance": True,
            "observation_goal": "show ordinary access around the shelf",
        }
    ]
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        )
    ]
    discovery["unusual_unconfirmed"] = [
        {
            "discovery_id": "lamp_confirmation",
            "target_ids": ["lamp"],
            "owning_group_id": "lighting",
            "observation_goal": "show the local operating affordance",
        }
    ]

    ledger = build_functional_check_ledger(discovery, groups=GROUPS)
    routed_sources = {
        source_id
        for check in ledger["checks"]
        for source_id in check["source_discovery_ids"]
    }

    assert {
        "tv_surface",
        "shelf_boundary",
        "shelf_clearance",
        "tv_sofa",
        "lamp_confirmation",
    } <= routed_sources
    assert all(
        check["owner_stage"]
        in {"cross_group_relation", "group_local"}
        for check in ledger["checks"]
    )


def test_singleton_confirmation_reuses_existing_directed_check() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": False,
            "observation_goal": "show the accessible shelf frontage",
        }
    ]
    discovery["unusual_unconfirmed"] = [
        {
            "discovery_id": "bookshelf_confirmation",
            "target_ids": ["bookshelf"],
            "owning_group_id": "storage",
            "observation_goal": "confirm the local access affordance",
        }
    ]

    ledger = build_functional_check_ledger(discovery, groups=GROUPS)
    checks = checks_for_group(ledger, "storage")

    assert len(checks) == 1
    assert checks[0]["check_type"] == "architecture_orientation"
    assert checks[0]["source_discovery_ids"] == [
        "bookshelf_surface",
        "bookshelf_confirmation",
    ]
    assert "confirm the local access affordance" in checks[0][
        "observation_goals"
    ]


def test_n021_rendered_relation_is_not_observation_complete() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "tv_surface",
            "target_id": "television",
            "owning_group_id": "media",
            "surface_roles": ["display_side"],
            "need_clearance": False,
            "observation_goal": "show the display side",
        },
        {
            "discovery_id": "sofa_surface",
            "target_id": "sofa",
            "owning_group_id": "seating",
            "surface_roles": ["seating_side"],
            "need_clearance": False,
            "observation_goal": "show the seating side",
        },
    ]
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        )
    ]
    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=4,
        groups=GROUPS,
    )
    relation_unit = next(
        item
        for item in plan["probe_units"]
        if item["route_scope"] == "cross_group"
    )
    relation_check = next(
        item
        for item in plan["functional_check_ledger"]["checks"]
        if item["check_family"] == "cross_group_correspondence"
    )
    probe_result = {
        **deepcopy(relation_unit),
        "status": "available",
        "evidence_paths": ["/tmp/tv_sofa.png"],
        "usable_surface_audit": {"hypotheses": []},
        "functional_geometry": {"surface_observations": []},
    }
    ledger = update_functional_check_evidence(
        plan["functional_check_ledger"],
        probe_results=[probe_result],
    )
    updated = next(
        item
        for item in ledger["checks"]
        if item["check_id"] == relation_check["check_id"]
    )
    packet = functional_relation_judge_packet(
        global_paths=["/tmp/global.png"],
        probe_result=probe_result,
        required_check=updated,
    )

    assert updated["artifact_rendered"] is True
    assert updated["observation_complete"] is False
    assert packet["artifact_rendered"] is True
    assert packet["machine_observation_complete"] is False
    assert packet["observation_complete"] is False
    assert packet["coverage_complete"] is False
    assert {
        item["target_id"] for item in updated["target_affordances"]
    } == {"television", "sofa"}


def test_affordance_provenance_does_not_claim_relation_acquisition() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "tv_surface",
            "target_id": "television",
            "owning_group_id": "media",
            "surface_roles": ["display_side"],
            "need_clearance": False,
            "observation_goal": "show the display side",
        }
    ]
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        )
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=2,
        groups=GROUPS,
    )
    relation_check = next(
        check
        for check in plan["functional_check_ledger"]["checks"]
        if check["check_family"] == "cross_group_correspondence"
    )
    frontage_check = next(
        check
        for check in plan["functional_check_ledger"]["checks"]
        if check["check_type"] == "architecture_orientation"
    )
    relation_unit = next(
        unit
        for unit in plan["probe_units"]
        if unit["route_scope"] == "cross_group"
    )
    frontage_unit = next(
        unit
        for unit in plan["probe_units"]
        if unit["route_scope"] == "group_local"
    )

    assert relation_check["source_discovery_ids"] == [
        "tv_sofa",
        "tv_surface",
    ]
    assert relation_check["routing_discovery_ids"] == ["tv_sofa"]
    assert relation_unit["check_ids"] == [relation_check["check_id"]]
    assert frontage_unit["check_ids"] == [frontage_check["check_id"]]


def test_check_acknowledgement_closes_the_audit_lifecycle() -> None:
    discovery = _base_discovery()
    discovery["within_group_correspondences"] = [
        {
            "discovery_id": "chair_table_correspondence",
            "target_ids": ["chair_1", "chair_2", "table"],
            "group_ids": ["dining"],
            "observation_kinds": ["mutual_orientation"],
            "observation_goal": "show chair and table interaction sides",
        }
    ]
    ledger = build_functional_check_ledger(
        discovery,
        groups=GROUPS,
    )
    required = checks_for_group(ledger, "dining")
    check = required[0]
    judgement = {
        "evidence_status": "sufficient",
        "verdict": "valid",
        "confidence": 0.8,
        "reason": "The interaction sides and relative facing are observable.",
        "missing_evidence": [],
        "defects": [],
        "evidence_request": None,
        "functional_check_results": [
            {
                "check_id": check["check_id"],
                "target_ids": check["target_ids"],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "All chairs present usable seating access to the table.",
            }
        ],
    }
    resolution = validate_functional_check_results(
        judgement,
        required_checks=required,
    )
    updated, coverage = apply_functional_check_judgements(
        ledger,
        relation_results=[],
        group_results=[
            {
                "group_id": "dining",
                "status": "evaluated",
                "judgement": judgement,
            }
        ],
    )

    assert resolution["complete"] is True
    assert coverage["complete"] is True
    assert updated["checks"][0]["lifecycle_status"] == "resolved"
    assert updated["checks"][0]["judge_result_ref"] == (
        "group_local_review:dining"
    )


def test_in_group_relation_uses_lazy_group_judge_acquisition() -> None:
    discovery = _base_discovery()
    discovery["within_group_correspondences"] = [
        {
            "discovery_id": "chair_table_correspondence",
            "target_ids": ["chair_1", "chair_2", "table"],
            "group_ids": ["dining"],
            "predicate": "directional_correspondence",
            "observation_goal": (
                "show chair interaction sides relative to the table"
            ),
        }
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=8,
        groups=GROUPS,
    )
    checks = checks_for_group(
        plan["functional_check_ledger"],
        "dining",
    )

    assert plan["probe_units"] == []
    assert plan["backfill_probe_units"] == []
    assert plan["budget_exhausted"] is False
    assert len(checks) == 1
    assert checks[0]["acquisition_policy"] == (
        "judge_requested_on_demand"
    )
    assert checks[0]["acquisition_status"] == (
        "judge_requested_on_demand"
    )
    assert plan["lazy_group_acquisition"] == {
        "policy": "required_check_driven_judge_requests_v1",
        "initial_evidence": [
            "one_angled_global_view",
            "one_group_local_view",
            "reused_existing_group_owned_probe_views",
        ],
        "required_check_ids": [checks[0]["check_id"]],
        "required_check_count": 1,
        "completion_condition": (
            "all_group_owned_required_checks_resolved_or_"
            "judge_episode_budget_exhausted"
        ),
        "decision_authority": "none",
    }


def test_group_invalid_row_keeps_acquisition_open_for_unresolved_check() -> None:
    discovery = _base_discovery()
    discovery["within_group_correspondences"] = [
        {
            "discovery_id": "chair_table_direction",
            "target_ids": ["chair_1", "chair_2", "table"],
            "group_ids": ["dining"],
            "predicate": "directional_correspondence",
            "observation_goal": "show chair directions relative to the table",
        },
        {
            "discovery_id": "chair_table_geometry",
            "target_ids": ["chair_1", "chair_2", "table"],
            "group_ids": ["dining"],
            "predicate": "relative_use_geometry",
            "observation_goal": "show joint-use distances around the table",
        },
    ]
    checks = checks_for_group(
        build_functional_check_ledger(discovery, groups=GROUPS),
        "dining",
    )
    invalid_check, unresolved_check = checks

    resolution = validate_functional_check_results(
        {
            "evidence_status": "insufficient",
            "verdict": "ambiguous",
            "confidence": 0.8,
            "reason": "One relation fails but another is not visible.",
            "missing_evidence": ["joint_visibility"],
            "defects": [],
            "evidence_request": {
                "target_ids": unresolved_check["target_ids"],
                "missing_observations": ["joint_visibility"],
                "view_goal": "show the unresolved joint-use geometry",
                "metadata": {
                    "unresolved_check_ids": [
                        unresolved_check["check_id"]
                    ]
                },
            },
            "functional_check_results": [
                {
                    "check_id": invalid_check["check_id"],
                    "target_ids": invalid_check["target_ids"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The visible relation is incompatible.",
                },
                {
                    "check_id": unresolved_check["check_id"],
                    "target_ids": unresolved_check["target_ids"],
                    "observation_status": "missing",
                    "conclusion": "unresolved",
                    "reason": "The interaction region is occluded.",
                },
            ],
        },
        required_checks=checks,
    )

    assert resolution["complete"] is False
    assert resolution["invalid_check_ids"] == [
        invalid_check["check_id"]
    ]
    assert resolution["unresolved_check_ids"] == [
        unresolved_check["check_id"]
    ]


def test_invalid_check_requires_defect_inside_its_object_scope() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": False,
            "observation_goal": "show the accessible shelf frontage",
        }
    ]
    required = checks_for_group(
        build_functional_check_ledger(discovery, groups=GROUPS),
        "storage",
    )
    check = required[0]
    judgement = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "A functional issue is visible.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "interaction_side_accessibility",
                "target_ids": ["lamp"],
                "relation": "unrelated object",
                "reason": "This does not support the bookshelf check.",
                "check_refs": [check["check_id"]],
            }
        ],
        "evidence_request": None,
        "functional_check_results": [
            {
                "check_id": check["check_id"],
                "target_ids": check["target_ids"],
                "observation_status": "observed",
                "conclusion": "invalid",
                "reason": "The shelf frontage is obstructed.",
            }
        ],
    }

    with pytest.raises(ValueError, match="object-level defect attribution"):
        validate_functional_check_results(
            judgement,
            required_checks=required,
        )


def test_invalid_check_requires_explicit_defect_check_reference() -> None:
    discovery = _base_discovery()
    discovery["within_group_correspondences"] = [
        {
            "discovery_id": "chair_table_direction",
            "target_ids": ["chair_1", "table"],
            "group_ids": ["dining"],
            "predicate": "directional_correspondence",
            "observation_goal": "show chair direction relative to the table",
        }
    ]
    check = checks_for_group(
        build_functional_check_ledger(discovery, groups=GROUPS),
        "dining",
    )[0]

    with pytest.raises(ValueError, match="check_refs linkage"):
        validate_functional_check_results(
            {
                "verdict": "invalid",
                "defects": [
                    {
                        "scope": "functional_relation",
                        "target_ids": ["chair_1"],
                        "relation": check["relation"],
                        "reason": "The chair faces away from the table.",
                    }
                ],
                "functional_check_results": [
                    {
                        "check_id": check["check_id"],
                        "target_ids": check["target_ids"],
                        "observation_status": "observed",
                        "conclusion": "invalid",
                        "reason": "The required direction is incompatible.",
                    }
                ],
            },
            required_checks=[check],
        )


def test_group_judge_may_find_a_new_defect_after_resolving_checks_valid() -> None:
    discovery = _base_discovery()
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "table_clearance",
            "target_id": "table",
            "owning_group_id": "dining",
            "need_clearance": True,
            "observation_goal": "show the table approach region",
        }
    ]
    required = checks_for_group(
        build_functional_check_ledger(discovery, groups=GROUPS),
        "dining",
    )
    check = required[0]
    resolution = validate_functional_check_results(
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "A separate local operating defect is visible.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "orientation_for_use",
                    "target_ids": ["chair_1"],
                    "relation": "seat faces away from the shared-use area",
                    "reason": "The chair's seating side faces away.",
                }
            ],
            "evidence_request": None,
            "functional_check_results": [
                {
                    "check_id": check["check_id"],
                    "target_ids": check["target_ids"],
                    "observation_status": "observed",
                    "conclusion": "valid",
                    "reason": "The table approach region itself is accessible.",
                }
            ],
        },
        required_checks=required,
    )

    assert resolution["complete"] is True
    assert resolution["invalid_check_ids"] == []


def test_relation_episode_invalid_must_map_to_its_required_check() -> None:
    discovery = _base_discovery()
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            "tv_sofa",
            "television",
            "sofa",
            "media",
            "seating",
        )
    ]
    check = next(
        item
        for item in build_functional_check_ledger(
            discovery,
            groups=GROUPS,
        )["checks"]
        if item["check_family"] == "cross_group_correspondence"
    )
    judgement = {
        "evidence_status": "sufficient",
        "verdict": "invalid",
        "confidence": 0.8,
        "reason": "The relation is defective.",
        "missing_evidence": [],
        "defects": [
            {
                "scope": "orientation_for_use",
                "target_ids": ["television"],
                "relation": "faces away from seating",
                "reason": "The display side faces away from the seating.",
            }
        ],
        "evidence_request": None,
        "functional_check_results": [
            {
                "check_id": check["check_id"],
                "target_ids": check["target_ids"],
                "observation_status": "observed",
                "conclusion": "valid",
                "reason": "This row conflicts with the relation verdict.",
            }
        ],
    }

    with pytest.raises(ValueError, match="requires an invalid verdict"):
        validate_functional_check_results(
            judgement,
            required_checks=[check],
            invalid_verdict_requires_invalid_check=True,
        )


def test_orientation_and_clearance_require_separate_rows_but_can_share_defect() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": "bookshelf_surface",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "surface_roles": ["access_side"],
            "need_clearance": True,
            "observation_goal": "show the usable frontage",
        }
    ]
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "bookshelf_clearance",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "need_clearance": True,
            "observation_goal": "show the frontage free-space region",
        }
    ]
    required = checks_for_group(
        build_functional_check_ledger(discovery, groups=GROUPS),
        "storage",
    )

    assert {item["check_type"] for item in required} == {
        "architecture_orientation",
        "clearance",
    }
    resolution = validate_functional_check_results(
        {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.9,
            "reason": "The same authored frontage failure affects both tests.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "interaction_side_accessibility",
                    "target_ids": ["bookshelf"],
                    "relation": "frontage faces and is blocked by boundary",
                    "reason": "The usable frontage cannot support ordinary use.",
                    "check_refs": [
                        check["check_id"] for check in required
                    ],
                }
            ],
            "evidence_request": None,
                "functional_check_results": [
                    {
                        "check_id": check["check_id"],
                        "target_ids": check["target_ids"],
                        "observation_status": "observed",
                        "conclusion": "invalid",
                        "reason": f"{check['check_type']} fails in the authored pose.",
                        **(
                            {
                                "affected_object_ids": ["bookshelf"],
                                "cause_kind": "self_layout",
                                "causal_object_ids": ["bookshelf"],
                                "scoring_target_ids": ["bookshelf"],
                            }
                            if check["check_type"] == "clearance"
                            else {}
                        ),
                    }
                    for check in required
                ],
        },
        required_checks=required,
    )

    assert set(resolution["invalid_check_ids"]) == {
        item["check_id"] for item in required
    }


def test_clearance_causal_shortlist_is_stable_bounded_and_not_a_whitelist() -> None:
    discovery = _base_discovery()
    discovery["approach_clearance_targets"] = [
        {
            "discovery_id": "bookshelf_clearance",
            "target_id": "bookshelf",
            "owning_group_id": "storage",
            "need_clearance": True,
            "observation_goal": "show the space around the bookshelf",
        }
    ]
    positions = {
        "bookshelf": ([0.0, 0.0, 0.5], [1.0, 0.4, 1.0]),
        "chair_1": ([0.4, 0.0, 0.5], [0.5, 0.5, 1.0]),
        "chair_2": ([1.0, 0.0, 0.5], [0.5, 0.5, 1.0]),
        "table": ([2.0, 0.0, 0.5], [0.8, 0.8, 1.0]),
        "television": ([3.0, 0.0, 0.5], [0.8, 0.2, 1.0]),
        "sofa": ([4.0, 0.0, 0.5], [1.6, 0.8, 1.0]),
        "lamp": ([5.0, 0.0, 0.5], [0.3, 0.3, 1.0]),
    }
    scene = {
        "objects": [
            {"id": object_id, "center": center, "size": size}
            for object_id, (center, size) in positions.items()
        ]
    }

    ledger = build_functional_check_ledger(
        discovery,
        groups=GROUPS,
        scene=scene,
    )
    check = ledger["checks"][0]

    assert check["causal_candidate_ids"][0] == "chair_1"
    assert len(check["causal_candidate_ids"]) == 4
    assert check["causal_candidates_are_routing_prior"] is True
    assert check["allowed_causal_object_ids"] == sorted(positions)
    # The lamp is deliberately outside the top-four routing prior. It remains
    # legal because nearby AABBs prioritize context but never whitelist it.
    resolution = validate_functional_check_results(
        {
            "verdict": "invalid",
            "defects": [
                {
                    "scope": "clearance",
                    "target_ids": ["lamp"],
                    "relation": "clearance",
                    "reason": "The lamp is the observed external blocker.",
                    "check_refs": [check["check_id"]],
                }
            ],
            "functional_check_results": [
                {
                    "check_id": check["check_id"],
                    "target_ids": ["bookshelf"],
                    "observation_status": "inferred_under_budget",
                    "conclusion": "invalid",
                    "reason": (
                        "The terminal evidence identifies the lamp as the "
                        "external blocker."
                    ),
                    "affected_object_ids": ["bookshelf"],
                    "cause_kind": "external_object",
                    "causal_object_ids": ["lamp"],
                    "scoring_target_ids": ["lamp"],
                }
            ],
        },
        required_checks=[check],
    )
    assert resolution["complete"] is True


def test_orientation_packet_binds_angled_global_and_reused_side_view() -> None:
    packet = {
        "required_checks": [
            {
                "check_id": "orientation",
                "check_type": "architecture_orientation",
                "target_ids": ["bookshelf"],
                "evidence_refs": ["/tmp/bookshelf_side.png"],
            },
            {
                "check_id": "clearance",
                "check_type": "clearance",
                "target_ids": ["bookshelf"],
                "evidence_refs": ["/tmp/bookshelf_side.png"],
            },
        ],
        "image_order": [
            {
                "artifact_id": "/tmp/bookshelf_side.png",
                "check_ids": ["orientation", "clearance"],
            }
        ],
    }

    _bind_architecture_orientation_evidence(
        packet,
        packet_paths=[
            "/tmp/scene_angled.png",
            "/tmp/bookshelf_side.png",
        ],
        angled_global_paths=["/tmp/scene_angled.png"],
    )

    policy = packet["architecture_orientation_policy"]
    binding = policy["evidence_bindings"][0]
    assert binding["angled_global_image_alias"] == "image_00"
    assert binding["side_conditioned_local_image_alias"] == "image_01"
    assert binding["reused_by_clearance_check_ids"] == ["clearance"]
    assert binding["status"] == "complete"
    assert policy["need_more_evidence_loop"] == "controller_managed"


def test_per_check_packet_excludes_other_object_probes() -> None:
    check = {
        "check_id": "chair_01_orientation",
        "check_type": "architecture_orientation",
        "target_ids": ["chair_01"],
        "target_affordances": [
            {
                "target_id": "chair_01",
                "directionality": "directed",
            }
        ],
        "required_observations": [
            "interaction_side_visible",
            "front_back_disambiguated",
        ],
    }
    packet = {
        "paths": [
            "/tmp/global.png",
            "/tmp/group.png",
            "/tmp/chair_01.png",
            "/tmp/chair_02.png",
        ],
        "resolution": {
            "functional_probe_reuse": {
                "baseline_packet_paths": [
                    "/tmp/global.png",
                    "/tmp/group.png",
                ],
                "requested_probe_paths": [
                    "/tmp/chair_01.png",
                    "/tmp/chair_02.png",
                ],
            }
        },
        "functional_probe_evidence": {
            "required_checks": [check],
            "image_order": [
                {
                    "artifact_id": "/tmp/chair_01.png",
                    "check_ids": ["chair_01_orientation"],
                    "target_ids": ["chair_01"],
                },
                {
                    "artifact_id": "/tmp/chair_02.png",
                    "check_ids": ["chair_02_orientation"],
                    "target_ids": ["chair_02"],
                },
            ],
        },
    }

    scoped = _scope_functional_episode_evidence(packet, check=check)

    assert scoped["paths"] == [
        "/tmp/global.png",
        "/tmp/group.png",
        "/tmp/chair_01.png",
    ]
    audit = scoped["resolution"]["functional_check_evidence_scope"]
    assert audit["omitted_unrelated_paths"] == ["/tmp/chair_02.png"]


def test_per_check_packets_scope_shared_measurements_to_one_check() -> None:
    checks = [
        {
            "check_id": "functional_check_001",
            "check_type": "architecture_orientation",
            "target_ids": ["chair_01"],
        },
        {
            "check_id": "functional_check_002",
            "check_type": "clearance",
            "target_ids": ["cabinet_01"],
        },
    ]
    shared_measurements = {
        "schema_version": "functional_measurement_bank_v1",
        "status": "complete",
        "measurement_role": (
            "deterministic_spatial_evidence_not_verdict"
        ),
        "decision_authority": "none",
        "requested_check_ids": [item["check_id"] for item in checks],
        "check_measurements": [
            {
                "check_id": item["check_id"],
                "check_type": item["check_type"],
                "target_ids": item["target_ids"],
                "status": "complete",
            }
            for item in checks
        ],
    }
    expanded = _expand_functional_check_packets(
        [
            {
                "paths": ["/tmp/global.png", "/tmp/group.png"],
                "resolution": {},
                "functional_probe_evidence": {
                    "required_checks": checks,
                    "functional_measurements": shared_measurements,
                },
            }
        ]
    )

    assert len(expanded) == 2
    for episode, expected_check in zip(expanded, checks, strict=True):
        functional = episode["functional_probe_evidence"]
        assert functional["required_check_ids"] == [
            expected_check["check_id"]
        ]
        assert [
            item["check_id"]
            for item in functional["functional_measurements"][
                "check_measurements"
            ]
        ] == [expected_check["check_id"]]


def test_missing_usable_side_requires_prejudge_acquisition() -> None:
    check = {
        "check_id": "orientation",
        "check_type": "architecture_orientation",
        "target_ids": ["bookshelf"],
        "target_affordances": [
            {
                "target_id": "bookshelf",
                "directionality": "directed",
            }
        ],
        "required_observations": [
            "interaction_side_visible",
            "front_back_disambiguated",
        ],
    }
    packet = {
        "paths": ["/tmp/global.png", "/tmp/group.png"],
        "functional_probe_evidence": {
            "required_checks": [check],
            "boundary_clearance_evidence": {
                "usable_surface_hypotheses": [],
            },
            "architecture_orientation_policy": {
                "evidence_bindings": [
                    {
                        "check_id": "orientation",
                        "status": "pending_more_evidence",
                    }
                ]
            },
        },
    }

    preflight = _functional_visual_preflight(
        packet,
        required_checks=[check],
    )

    assert preflight is not None
    assert preflight["target_ids"] == ["bookshelf"]
    assert preflight["reason_codes"] == [
        "usable_surface_not_machine_resolved",
        "side_conditioned_view_not_bound",
    ]


def test_directional_relation_is_rejudged_after_endpoint_conflict() -> None:
    relation_check = {
        "check_id": "relation",
        "check_type": "directional_correspondence",
        "target_ids": ["sofa", "television"],
    }
    orientation_check = {
        "check_id": "orientation",
        "check_type": "architecture_orientation",
        "target_ids": ["television"],
    }
    spec = {
        "relation_id": "sofa_tv",
        "target_ids": ["sofa", "television"],
        "group_ids": ["seating", "media"],
        "groups": [],
        "relation_predicates": ["directional_correspondence"],
        "observation_kinds": ["directional_correspondence"],
        "observation_goals": ["inspect joint viewing orientation"],
        "evidence_paths": ["/tmp/relation.png"],
        "pair_specific_evidence_available": True,
        "required_checks": [relation_check],
        "required_check_ids": ["relation"],
        "required_check_id": "relation",
        "judge_packet": {"required_checks": [relation_check]},
    }
    relation_result = {
        "relation_id": "sofa_tv",
        "status": "evaluated",
        "score": 1.0,
        "vlm_invoked": True,
        "judge_episode_count": 1,
        "judgement": {
            "verdict": "valid",
            "functional_check_results": [
                {
                    "check_id": "relation",
                    "target_ids": ["sofa", "television"],
                    "observation_status": "observed",
                    "conclusion": "valid",
                    "reason": "The pair appears mutually oriented.",
                }
            ],
        },
    }
    group_result = {
        "group_id": "media",
        "check_episodes": [
            {
                "group_id": "media",
                "evidence_paths": ["/tmp/tv_side.png"],
                "functional_probe_evidence": {
                    "required_checks": [orientation_check]
                },
                "judgement": {
                    "functional_check_results": [
                        {
                            "check_id": "orientation",
                            "target_ids": ["television"],
                            "observation_status": "observed",
                            "conclusion": "invalid",
                            "reason": "The display side faces the boundary.",
                        }
                    ]
                },
            }
        ],
    }
    requests: list[dict] = []

    def call_judge(_judge, request):
        requests.append(deepcopy(request))
        return {
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "The combined evidence shows incompatible facing.",
            "missing_evidence": [],
            "evidence_request": None,
            "functional_check_results": [
                {
                    "check_id": "relation",
                    "target_ids": ["sofa", "television"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The display and seating sides do not align.",
                }
            ],
            "defects": [
                {
                    "scope": "functional_correspondence",
                    "target_ids": ["television"],
                    "relation": "directional_correspondence",
                    "reason": "The television faces away from the sofa.",
                    "category": "functional_correspondence_failure",
                    "severity": "impaired",
                    "attribution_mode": "responsible_endpoint",
                    "check_refs": ["relation"],
                }
            ],
        }

    reconciled, audit = reconcile_directional_relation_conflicts(
        specs=[spec],
        relation_results=[relation_result],
        group_results=[group_result],
        metric_name="functional_consistency",
        scene={"objects": []},
        global_evidence=["/tmp/global.png"],
        vlm_judge=object(),
        prompt=None,
        visual_style_spec=None,
        authorized_deviations=[],
        build_judge_request=lambda **kwargs: kwargs,
        call_judge=call_judge,
        apply_prompt_exemptions=lambda value, **_: value,
        normalize_judgement=lambda value, **_: {
            "status": "evaluated",
            "score": 0.0 if value["verdict"] == "invalid" else 1.0,
            "reason": None,
        },
    )

    assert requests[0]["render_evidence"] == [
        "/tmp/global.png",
        "/tmp/relation.png",
        "/tmp/tv_side.png",
    ]
    assert reconciled[0]["score"] == 0.0
    assert reconciled[0]["judge_episode_count"] == 2
    assert reconciled[0]["judgement"]["defects"][0][
        "same_physical_event_check_ref"
    ] == "orientation"
    assert reconciled[0]["same_physical_event_deduplication"][
        "link_count"
    ] == 1
    assert audit["conflict_count"] == 1


def test_empty_functional_inventory_preserves_non_regression_path() -> None:
    discovery = _base_discovery()
    ledger = build_functional_check_ledger(
        discovery,
        groups=GROUPS,
    )

    assert ledger["checks"] == []
    assert validate_functional_check_results(
        {"verdict": "valid"},
        required_checks=[],
    )["complete"] is not False
