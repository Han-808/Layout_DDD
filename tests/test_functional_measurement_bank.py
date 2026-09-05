from __future__ import annotations

import json

import numpy as np
import pytest

from benchmark.evaluator.scene_quality.functional_acquisition import (
    build_functional_acquisition_plan,
)
from benchmark.evaluator.scene_quality.functional_measurements import (
    attach_functional_measurement_extension,
    build_functional_measurement_bank,
    compact_functional_measurements_for_checks,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    functional_probe_judge_packet,
)
from benchmark.rendering.camera_pose import (
    _CameraObject,
    _functional_surface_observability,
)


def _scene() -> dict:
    return {
        "scene_id": "measurement_scene",
        "boundary": [[0, 0], [6, 0], [6, 6], [0, 6]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "chair",
                "category": "chair",
                "center": [1.0, 3.0, 0.5],
                "size": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            {
                "id": "monitor",
                "category": "monitor",
                "center": [5.0, 3.0, 1.0],
                "size": [1.0, 0.2, 1.0],
                "rotation": [0.0, 0.0, 0.0],
            },
        ],
    }


def _discovery() -> dict:
    return {
        "inspected_object_ids": ["chair", "monitor"],
        "directed_surface_targets": [
            {
                "discovery_id": "chair_side",
                "target_id": "chair",
                "directionality": "directed",
                "surface_roles": ["seating_side"],
                "need_clearance": False,
                "owning_group_id": "g",
                "observation_goal": "confirm the chair seating direction",
                "precomputed_usable_surface_hypothesis": {
                    "target_id": "chair",
                    "status": "identified",
                    "surfaces": [
                        {
                            "surface_role": "seating_side",
                            "side_id": "local_pos_x",
                            "confidence": 0.9,
                        }
                    ],
                },
            },
            {
                "discovery_id": "monitor_side",
                "target_id": "monitor",
                "directionality": "directed",
                "surface_roles": ["display_side"],
                "need_clearance": False,
                "owning_group_id": "g",
                "observation_goal": "confirm the monitor display direction",
                "precomputed_usable_surface_hypothesis": {
                    "target_id": "monitor",
                    "status": "identified",
                    "surfaces": [
                        {
                            "surface_role": "display_side",
                            "side_id": "local_neg_x",
                            "confidence": 0.95,
                        }
                    ],
                },
            },
        ],
        "within_group_correspondences": [
            {
                "discovery_id": "chair_monitor_relation",
                "target_ids": ["chair", "monitor"],
                "group_ids": ["g"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "confirm direct viewing correspondence",
            }
        ],
        "cross_group_correspondences": [],
        "approach_clearance_targets": [],
        "boundary_sensitive_targets": [],
        "unusual_unconfirmed": [],
    }


def test_measurement_bank_covers_all_checks_before_zero_budget_scheduling() -> None:
    plan = build_functional_acquisition_plan(
        _discovery(),
        max_probe_units=0,
        groups=[{"group_id": "g", "object_ids": ["chair", "monitor"]}],
        scene=_scene(),
    )

    ledger = plan["functional_check_ledger"]
    bank = plan["functional_measurement_bank"]
    assert plan["probe_units"] == []
    assert bank["generation_stage"] == (
        "accepted_checks_before_camera_scheduling"
    )
    assert bank["accepted_check_count"] == len(ledger["checks"]) == 3
    assert bank["check_measurement_count"] == 3
    assert bank["coverage"]["record_coverage_complete"] is True
    assert {
        row["check_id"] for row in bank["check_measurements"]
    } == {check["check_id"] for check in ledger["checks"]}

    relation_check = next(
        check
        for check in ledger["checks"]
        if check["check_type"] == "directional_correspondence"
    )
    relation_measurement = next(
        row
        for row in bank["check_measurements"]
        if row["check_id"] == relation_check["check_id"]
    )
    chair_to_monitor = next(
        row
        for row in relation_measurement["pair_measurements"]
        if row["endpoint_id"] == "chair"
    )
    assert chair_to_monitor["center_distance_m"] == pytest.approx(4.0)
    assert chair_to_monitor["surface_relations"][0][
        "facing_angle_to_counterpart_degrees"
    ] == pytest.approx(0.0)
    assert relation_measurement["provenance"][
        "camera_schedule_consulted"
    ] is False

    compact = compact_functional_measurements_for_checks(
        bank,
        [relation_check["check_id"]],
    )
    serialized = json.dumps(compact)
    assert compact["status"] == "complete"
    assert "center_xyz" not in serialized
    assert "size_xyz" not in serialized
    assert "world_outward_direction" in serialized

    cross_group_check = dict(relation_check)
    cross_group_check["owner_stage"] = "cross_group_relation"
    packet = functional_probe_judge_packet(
        global_paths=[],
        probe_paths=[],
        acquisition_audit={
            "functional_check_ledger": {"checks": [cross_group_check]},
            "functional_measurement_bank": bank,
            "probe_results": [],
        },
    )
    assert packet["functional_measurements"] == compact


def test_measurement_extension_is_namespaced_and_cannot_add_a_verdict() -> None:
    plan = build_functional_acquisition_plan(
        _discovery(),
        max_probe_units=0,
        groups=[{"group_id": "g", "object_ids": ["chair", "monitor"]}],
        scene=_scene(),
    )
    bank = plan["functional_measurement_bank"]
    check_id = bank["check_measurements"][0]["check_id"]

    extended = attach_functional_measurement_extension(
        bank,
        namespace="directional_clearance",
        by_check_id={
            check_id: {
                "status": "available",
                "nearest_forward_obstacle_distance_m": 1.25,
            }
        },
        source="deterministic_directional_clearance_v1",
    )
    payload = extended["check_measurements"][0][
        "measurement_extensions"
    ]["directional_clearance"]
    assert payload["measurement"][
        "nearest_forward_obstacle_distance_m"
    ] == pytest.approx(1.25)
    compact = compact_functional_measurements_for_checks(
        extended,
        [check_id],
    )
    assert compact["check_measurements"][0]["measurement_extensions"][
        "directional_clearance"
    ] == payload
    assert bank["check_measurements"][0]["measurement_extensions"] == {}

    with pytest.raises(ValueError, match="decision fields"):
        attach_functional_measurement_extension(
            bank,
            namespace="bad",
            by_check_id={check_id: {"verdict": "invalid"}},
            source="test",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("center", None),
        ("center", [float("nan"), 3.0, 0.5]),
        ("size", [1.0, 0.0, 1.0]),
    ],
)
def test_measurement_bank_never_fabricates_zero_geometry(
    field: str,
    value: object,
) -> None:
    scene = _scene()
    scene["objects"][0][field] = value
    ledger = {
        "checks": [
            {
                "check_id": "functional_check_001",
                "check_type": "directional_correspondence",
                "owner_stage": "group_local",
                "target_ids": ["chair", "monitor"],
            }
        ]
    }

    bank = build_functional_measurement_bank(
        scene=scene,
        functional_check_ledger=ledger,
        discovery=_discovery(),
    )

    row = bank["check_measurements"][0]
    chair = next(
        item
        for item in row["target_measurements"]
        if item["target_id"] == "chair"
    )
    chair_pair = next(
        item
        for item in row["pair_measurements"]
        if item["endpoint_id"] == "chair"
        and item["counterpart_id"] == "monitor"
    )
    assert row["status"] == "partial"
    assert chair["geometry_status"] == "unavailable"
    assert "center_xyz" not in chair
    assert "size_xyz" not in chair
    assert chair["usable_surfaces"] == []
    if field == "center":
        assert chair_pair["geometry_status"] == "unavailable"
        assert "center_distance_m" not in chair_pair
    assert any(
        item["field"] == "targets.chair.scene_geometry"
        for item in row["unavailable_fields"]
    )


def test_invalid_rotation_cannot_create_a_fake_world_heading() -> None:
    scene = _scene()
    scene["objects"][0]["rotation"] = [0.0, 0.0, float("inf")]
    ledger = {
        "checks": [
            {
                "check_id": "functional_check_001",
                "check_type": "architecture_orientation",
                "owner_stage": "group_local",
                "target_ids": ["chair"],
            }
        ]
    }

    row = build_functional_measurement_bank(
        scene=scene,
        functional_check_ledger=ledger,
        discovery=_discovery(),
    )["check_measurements"][0]

    assert row["status"] == "partial"
    assert row["target_measurements"][0]["usable_surfaces"] == []
    assert any(
        item["field"] == "targets.chair.rotation"
        for item in row["unavailable_fields"]
    )
    assert not any(
        "world_outward_direction" in str(item)
        for item in row["target_measurements"]
    )


def _camera_object() -> _CameraObject:
    return _CameraObject(
        id="chair",
        center=np.asarray([0.0, 0.0, 0.5]),
        half=np.asarray([0.5, 0.5, 0.5]),
        R=np.eye(3),
        bottom_z=0.0,
        top_z=1.0,
    )


def _surface_probe(*, directional_relation: bool) -> dict:
    return {
        "kind": (
            "functional_correspondence"
            if directional_relation
            else "functional_frontage"
        ),
        "relation_predicates": (
            ["directional_correspondence"] if directional_relation else []
        ),
        "usable_surface_hypotheses": [
            {
                "target_id": "chair",
                "status": "identified",
                "surfaces": [{"side_id": "local_pos_y"}],
            }
        ],
    }


def test_predicate_aware_surface_coverage_keeps_feasible_profile_views() -> None:
    obj = _camera_object()
    front = _functional_surface_observability(
        _surface_probe(directional_relation=False),
        objects=[obj],
        camera_location=np.asarray([0.0, 2.0, 1.0]),
    )
    profile = _functional_surface_observability(
        _surface_probe(directional_relation=False),
        objects=[obj],
        camera_location=np.asarray([2.0, 0.0, 1.0]),
    )
    rear = _functional_surface_observability(
        _surface_probe(directional_relation=False),
        objects=[obj],
        camera_location=np.asarray([0.0, -2.0, 1.0]),
    )

    assert front["coverage_status"] == "sufficient"
    assert profile["coverage_status"] == "partial_but_usable"
    assert profile["eligible"] is True
    assert rear["coverage_status"] == "not_covered"
    assert rear["eligible"] is False

    # A slightly rearward joint profile remains usable for a directional
    # relation because the Judge also receives deterministic endpoint heading.
    directional_profile = _functional_surface_observability(
        _surface_probe(directional_relation=True),
        objects=[obj],
        camera_location=np.asarray([1.0, -0.15, 1.0]),
    )
    frontage_same_pose = _functional_surface_observability(
        _surface_probe(directional_relation=False),
        objects=[obj],
        camera_location=np.asarray([1.0, -0.15, 1.0]),
    )
    assert directional_profile["coverage_status"] == "partial_but_usable"
    assert frontage_same_pose["coverage_status"] == "not_covered"


def test_missing_side_hypothesis_is_partial_instead_of_false_sufficient() -> None:
    result = _functional_surface_observability(
        {
            "kind": "functional_frontage",
            "surface_targets": [{"target_id": "chair"}],
            "usable_surface_hypotheses": [],
        },
        objects=[_camera_object()],
        camera_location=np.asarray([0.0, 2.0, 1.0]),
    )

    assert result["coverage_status"] == "partial_but_usable"
    assert result["eligible"] is True
    assert result["required_target_ids"] == ["chair"]
    assert result["partial_hypotheses"] == [
        {
            "target_id": "chair",
            "side_id": None,
            "coverage_status": "partial_but_usable",
            "reason_code": "usable_side_hypothesis_not_available",
            "surface_hypothesis_status": "missing",
        }
    ]
