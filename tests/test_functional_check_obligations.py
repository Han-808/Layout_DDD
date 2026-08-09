from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.evaluator.scene_quality.functional_acquisition import (
    build_functional_acquisition_plan,
)
from benchmark.evaluator.scene_quality.functional_checks import (
    apply_functional_check_judgements,
    build_functional_check_ledger,
    canonicalize_typed_invalid_envelope,
    checks_for_group,
    forced_group_ids_from_checks,
    update_functional_check_evidence,
    validate_functional_check_results,
)
from benchmark.evaluator.scene_quality.cross_group_relations import (
    _cross_group_relation_episode_specs,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    functional_relation_judge_packet,
)
from benchmark.evaluator.scene_quality.global_group_first import (
    _bind_architecture_orientation_evidence,
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


def test_n011_scheduler_is_fair_to_usable_side_family() -> None:
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
    assert any(
        item.get("owning_group_id") == "storage"
        and "bookshelf_approach" in item.get("discovery_ids", [])
        for item in plan["probe_units"]
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
        for item in plan["probe_units"]
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


def test_scheduler_prefers_new_directed_objects_over_repeated_relations() -> None:
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
    ) == 1
    selected_targets = {
        target_id
        for unit in plan["probe_units"]
        for target_id in [
            *unit.get("target_ids", []),
            *unit.get("related_target_ids", []),
        ]
    }
    assert {"bookshelf", "lamp"} <= selected_targets
    directed_gains = {
        target_id
        for unit in plan["probe_units"]
        for target_id in unit["scheduling_coverage_gain"][
            "new_directed_object_ids"
        ]
    }
    assert {"bookshelf", "lamp"} <= directed_gains


def test_functional_probe_budget_is_hard_capped_at_eight() -> None:
    discovery = _base_discovery()
    discovery["directed_surface_targets"] = [
        {
            "discovery_id": f"surface_{object_id}",
            "target_id": object_id,
            "owning_group_id": next(
                group["group_id"]
                for group in GROUPS
                if object_id in group["object_ids"]
            ),
            "surface_roles": ["interaction_side"],
            "need_clearance": False,
            "observation_goal": f"show {object_id} interaction side",
        }
        for object_id in discovery["inspected_object_ids"]
    ]
    discovery["cross_group_correspondences"] = [
        _cross_relation(
            f"relation_{target}",
            "television",
            target,
            "media",
            group_id,
        )
        for target, group_id in (
            ("sofa", "seating"),
            ("chair_1", "dining"),
            ("bookshelf", "storage"),
            ("lamp", "lighting"),
        )
    ]

    plan = build_functional_acquisition_plan(
        discovery,
        max_probe_units=99,
        groups=GROUPS,
    )

    assert plan["max_probe_units"] == FUNCTIONAL_PROBE_MAX_UNITS == 8
    assert len(plan["probe_units"]) == 8
    assert plan["budget"]["max_probe_units"] == {
        "requested": 99,
        "effective": 8,
        "hard_cap": 8,
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


def test_group_final_invalid_preserves_other_unresolved_check_for_audit() -> None:
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
            "evidence_status": "sufficient",
            "verdict": "invalid",
            "confidence": 0.8,
            "reason": "One relation fails but another is not visible.",
            "missing_evidence": [],
            "defects": [
                {
                    "scope": "functional_relation",
                    "target_ids": invalid_check["target_ids"],
                    "relation": invalid_check["relation"],
                    "reason": "The visible relation is incompatible.",
                    "check_refs": [invalid_check["check_id"]],
                }
            ],
            "evidence_request": None,
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
