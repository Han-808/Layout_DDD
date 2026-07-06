from __future__ import annotations

from dataclasses import asdict

from benchmark.workflow.grouping import build_object_grouping_report, build_object_groups, resolve_grouping_config


def _obj(object_id: str, center: list[float], support_parent: str | None = None) -> dict:
    obj = {"object_id": object_id, "category": object_id.split("_")[0], "center": center, "size": [0.5, 0.5, 0.5], "yaw": 0}
    if support_parent:
        obj["support_parent"] = support_parent
    return obj


def _grouping_config(**overrides: object) -> dict:
    grouping = {
        "enabled": True,
        "proximity": {"min_gap_m": 0.25, "max_gap_m": 1.25, "max_normalized_gap": 0.75},
        "diameter": {"ratio_of_room_diagonal": 0.35, "min_m": 2.5, "max_m": 8.0},
        "object_count": {
            "rule": "sqrt_num_renderable_objects",
            "additive_margin": 1,
            "min_objects_per_group": 6,
            "max_objects_per_group": 12,
        },
        "strong_link_relaxation": {"max_group_diameter_multiplier": 1.25, "max_objects_multiplier": 1.5},
        "derived_support": {"enabled": True, "vertical_tolerance_m": 0.08, "min_xy_overlap_ratio": 0.15},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(grouping.get(key), dict):
            grouping[key] = {**grouping[key], **value}
        else:
            grouping[key] = value
    return {"grouping": grouping}


def test_grouping_assigns_every_object_once_and_splits_far_objects() -> None:
    layout = {"objects": [_obj("chair_1", [0, 0, 0.25]), _obj("desk_1", [0.8, 0, 0.25]), _obj("bed_1", [8, 8, 0.25])]}
    case = {"relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}]}

    groups = build_object_groups(layout, case, _grouping_config(proximity={"max_gap_m": 0.2}))
    assigned = [object_id for group in groups for object_id in group["object_ids"]]

    assert sorted(assigned) == ["bed_1", "chair_1", "desk_1"]
    assert len(assigned) == len(set(assigned))
    assert any({"chair_1", "desk_1"} == set(group["object_ids"]) for group in groups)
    assert any(group["object_ids"] == ["bed_1"] for group in groups)
    assert all(group["group_source"] == "spatial_cluster" for group in groups)


def test_grouping_prefers_semantic_regions_when_available() -> None:
    layout = {
        "objects": [
            _obj("chair_1", [0.2, 0.2, 0.25]),
            _obj("desk_1", [0.8, 0.2, 0.25]),
            _obj("bed_1", [8.0, 8.0, 0.25]),
            _obj("lamp_1", [8.5, 8.0, 0.25]),
        ]
    }
    case = {
        "room": {
            "floor_plan": {
                "regions": [
                    {"id": "work_0", "label": "work", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
                    {"id": "sleep_0", "label": "sleep", "floor_polygon": [[7, 7], [10, 7], [10, 10], [7, 10]]},
                ]
            }
        },
        "objects": [
            {"id": "chair_1", "source_region_id": "work_0"},
            {"id": "desk_1", "source_region_id": "work_0"},
            {"id": "bed_1", "source_region_id": "sleep_0"},
            {"id": "lamp_1", "source_region_id": "sleep_0"},
        ],
    }

    report = build_object_grouping_report(layout, case, _grouping_config(proximity={"max_gap_m": 0.0}))

    assert report["resolved_grouping_config"]["grouping_source"] == "semantic_region"
    assert {tuple(group["object_ids"]) for group in report["object_groups"]} == {
        ("bed_1", "lamp_1"),
        ("chair_1", "desk_1"),
    }
    assert all(group["group_source"] == "semantic_region" for group in report["object_groups"])
    assert {group["region_id"] for group in report["object_groups"]} == {"work_0", "sleep_0"}


def test_grouping_falls_back_when_region_assignment_is_sparse() -> None:
    layout = {"objects": [_obj("chair_1", [0, 0, 0.25]), _obj("desk_1", [0.8, 0, 0.25]), _obj("bed_1", [8, 8, 0.25])]}
    case = {"objects": [{"id": "chair_1", "source_region_id": "work_0"}]}

    report = build_object_grouping_report(layout, case, _grouping_config())

    assert report["resolved_grouping_config"].get("grouping_source") is None
    assert all(group["group_source"] == "spatial_cluster" for group in report["object_groups"])


def test_grouping_keeps_attachment_and_support_links_together() -> None:
    layout = {"objects": [_obj("desk_1", [0, 0, 0.4]), _obj("lamp_1", [0, 0, 0.9], support_parent="desk_1")]}
    case = {"attachments": [{"id": "att_1", "type": "support", "child": "lamp_1", "parent": "desk_1"}]}

    groups = build_object_groups(
        layout,
        case,
        _grouping_config(
            diameter={"ratio_of_room_diagonal": 0.01, "min_m": 0.1, "max_m": 0.1},
            object_count={"additive_margin": 0, "min_objects_per_group": 1, "max_objects_per_group": 1},
        ),
    )

    assert len(groups) == 1
    assert set(groups[0]["object_ids"]) == {"desk_1", "lamp_1"}
    assert any(reason.startswith("support_parent") or reason.startswith("attachment") for reason in groups[0]["edge_reasons"])
    assert any(
        edge["source"] == "desk_1"
        and edge["target"] == "lamp_1"
        and edge["reason"] == "support_parent"
        and edge["strength"] == "must_link"
        for edge in groups[0]["formation_edges"]
    )
    assert any(edge["reason"] == "attachment" and edge["strength"] == "must_link" for edge in groups[0]["formation_edges"])


def test_must_link_edges_are_not_broken_by_soft_limits() -> None:
    layout = {"objects": [_obj("base_1", [0, 0, 0.25])] + [_obj(f"item_{idx}", [idx, 0, 0.25], support_parent="base_1") for idx in range(8)]}

    groups = build_object_groups(
        layout,
        {},
        _grouping_config(
            diameter={"ratio_of_room_diagonal": 0.01, "min_m": 0.1, "max_m": 0.1},
            object_count={"additive_margin": 0, "min_objects_per_group": 1, "max_objects_per_group": 1},
        ),
    )

    assert len(groups) == 1
    assert len(groups[0]["object_ids"]) == 9
    assert sum(1 for edge in groups[0]["formation_edges"] if edge["strength"] == "must_link") == 8


def test_scale_aware_limits_increase_with_room_and_object_count() -> None:
    layout_44 = {"objects": [_obj(f"object_{idx:03d}", [idx % 11, idx // 11, 0.25]) for idx in range(44)]}
    layout_74 = {"objects": [_obj(f"object_{idx:03d}", [idx % 12, idx // 12, 0.25]) for idx in range(74)]}
    case = {"room": {"boundary": [[0, 0], [20, 0], [20, 10], [0, 10]]}}

    resolved_44 = resolve_grouping_config(layout_44, case, _grouping_config())
    resolved_74 = resolve_grouping_config(layout_74, case, _grouping_config())

    assert resolved_44.effective_max_group_diameter_m > 3.0
    assert resolved_44.effective_max_objects_per_group >= 8
    assert resolved_74.effective_max_objects_per_group >= 10
    assert resolved_74.effective_max_objects_per_group >= resolved_44.effective_max_objects_per_group


def test_scale_aware_proximity_gap_is_clamped() -> None:
    layout = {
        "objects": [
            {"object_id": "tiny_a", "category": "tiny", "center": [0, 0, 0.1], "size": [0.1, 0.1, 0.2], "yaw": 0},
            {"object_id": "tiny_b", "category": "tiny", "center": [0.34, 0, 0.1], "size": [0.1, 0.1, 0.2], "yaw": 0},
            {"object_id": "large_a", "category": "large", "center": [10, 0, 0.5], "size": [4, 4, 1], "yaw": 0},
            {"object_id": "large_b", "category": "large", "center": [15.1, 0, 0.5], "size": [4, 4, 1], "yaw": 0},
        ]
    }

    groups = build_object_groups(layout, {}, _grouping_config(diameter={"ratio_of_room_diagonal": 1.0, "min_m": 2.5, "max_m": 20.0}))

    assert any(set(group["object_ids"]) == {"tiny_a", "tiny_b"} for group in groups)
    assert any(set(group["object_ids"]) == {"large_a", "large_b"} for group in groups)


def test_strong_link_uses_relaxed_limits_and_records_edge_metadata() -> None:
    layout = {"objects": [_obj("chair_1", [0, 0, 0.25]), _obj("desk_1", [3.5, 0, 0.25])]}
    case = {
        "room": {"boundary": [[0, 0], [10, 0], [10, 4], [0, 4]]},
        "relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}],
    }
    config = _grouping_config(
        proximity={"min_gap_m": 0.25, "max_gap_m": 0.3, "max_normalized_gap": 0.1},
        diameter={"ratio_of_room_diagonal": 0.25, "min_m": 2.5, "max_m": 3.0},
        strong_link_relaxation={"max_group_diameter_multiplier": 1.5, "max_objects_multiplier": 1.5},
    )

    groups = build_object_groups(layout, case, config)

    group = next(group for group in groups if set(group["object_ids"]) == {"chair_1", "desk_1"})
    edge = group["formation_edges"][0]
    assert edge["reason"] == "explicit_relation"
    assert edge["strength"] == "strong_link"
    assert edge["priority"] == 1
    assert edge["is_ground_truth_relation"] is True
    assert edge["derived_from"] == "case.visible_relations"


def test_proximity_edges_can_be_rejected_by_group_limits() -> None:
    layout = {"objects": [_obj(f"box_{idx}", [idx * 0.45, 0, 0.25]) for idx in range(4)]}

    report = build_object_grouping_report(
        layout,
        {},
        _grouping_config(
            proximity={"min_gap_m": 0.0, "max_gap_m": 1.0, "max_normalized_gap": 1.0},
            diameter={"ratio_of_room_diagonal": 1.0, "min_m": 10.0, "max_m": 10.0},
            object_count={"additive_margin": 0, "min_objects_per_group": 2, "max_objects_per_group": 2},
        ),
    )

    assert report["omitted_edges"]
    assert any(edge["reason"] == "proximity" and edge["rejected_reason"] == "group_limit" for edge in report["omitted_edges"])
    assigned = [object_id for group in report["object_groups"] for object_id in group["object_ids"]]
    assert sorted(assigned) == [f"box_{idx}" for idx in range(4)]


def test_cross_group_explicit_relations_are_recorded_when_limits_reject_merge() -> None:
    layout = {"objects": [_obj("chair_1", [0, 0, 0.25]), _obj("desk_1", [8, 0, 0.25])]}
    case = {"relations": [{"id": "rel_1", "type": "near", "subject": "chair_1", "object": "desk_1"}]}

    report = build_object_grouping_report(
        layout,
        case,
        _grouping_config(
            diameter={"ratio_of_room_diagonal": 0.01, "min_m": 1.0, "max_m": 1.0},
            object_count={"additive_margin": 0, "min_objects_per_group": 6, "max_objects_per_group": 6},
            strong_link_relaxation={"max_group_diameter_multiplier": 1.0, "max_objects_multiplier": 1.0},
        ),
    )

    assert len(report["object_groups"]) == 2
    assert report["cross_group_relations"] == [
        {
            "source": "chair_1",
            "target": "desk_1",
            "reason": "explicit_relation",
            "source_group": "group_001",
            "target_group": "group_002",
            "status": "cross_group_due_to_limits",
        }
    ]


def test_derived_support_geometry_adds_non_ground_truth_strong_edge() -> None:
    layout = {
        "objects": [
            {"object_id": "table_1", "category": "table", "center": [0, 0, 0.5], "size": [2, 1, 1], "yaw": 0},
            {"object_id": "lamp_1", "category": "lamp", "center": [0, 0, 1.3], "size": [0.4, 0.4, 0.6], "yaw": 0},
        ]
    }

    groups = build_object_groups(layout, {}, _grouping_config())

    edge = groups[0]["formation_edges"][0]
    assert set(groups[0]["object_ids"]) == {"table_1", "lamp_1"}
    assert edge["reason"] == "derived_support_geometry"
    assert edge["strength"] == "strong_link"
    assert edge["is_ground_truth_relation"] is False


def test_resolved_grouping_config_does_not_expose_legacy_fallback_fields() -> None:
    layout = {"objects": [_obj("chair_1", [0, 0, 0.25]), _obj("desk_1", [0.8, 0, 0.25])]}

    resolved = resolve_grouping_config(
        layout,
        {},
        {
            "grouping": {
                "scale_aware": False,
                "legacy_fallback": {"max_group_diameter_m": 0.1, "max_objects_per_group": 1},
            }
        },
    )
    serialized = asdict(resolved)

    assert "scale_aware" not in serialized
    assert not any(key.startswith("legacy_") for key in serialized)
    assert resolved.effective_max_group_diameter_m >= 2.5
