from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest

from benchmark.functional_spatial_context import (
    project_functional_spatial_context,
)
from benchmark.visual_judge.placement_discovery import (
    discover_openai_compatible_placement_evidence,
)


class _Model:
    model_id = "model"
    endpoint = "https://example.test/v1"
    response_format_json = True

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []
        self.last_request_metadata: dict = {}

    def chat_messages(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


def _projection_audit(*, clearance: int, relations: int) -> dict:
    return {
        "policy": "bounded_attention_projection_v1",
        "max_context_chars": 22000,
        "eligible_clearance_count": clearance,
        "emitted_clearance_count": clearance,
        "eligible_relation_count": relations,
        "emitted_relation_count": relations,
        "reduced_measurement_object_ids": [],
        "omitted_measurement_object_ids": [],
        "omitted_clearance_object_ids": [],
        "omitted_relation_source_ids": [],
        "decision_authority": "none",
    }


def _functional_report() -> dict:
    return {
        "functional_discovery": {
            "schema_version": "functional_discovery_v5",
            "decision_authority": "none",
            "approach_clearance_targets": [
                {
                    "target_id": "wardrobe",
                    "need_clearance": True,
                    "observation_goal": "inspect ordinary opening clearance",
                    "discovery_id": "approach_clearance_01",
                },
                {
                    "target_id": "lamp",
                    "need_clearance": False,
                },
            ],
            "within_group_correspondences": [
                {
                    "target_ids": ["chair", "table"],
                    "predicate": "relative_use_geometry",
                    "dependency": "required",
                    "counterpart_mode": "dedicated",
                    "ordinary_mobility": "fixed",
                    "observation_goal": (
                        "inspect whether their relative distance permits "
                        "ordinary dining"
                    ),
                    "discovery_id": "functional_correspondence_01",
                    "scope": "within_group",
                },
                {
                    # Broad legacy clusters are not forwarded.
                    "target_ids": ["chair", "table", "lamp"],
                    "predicate": "relative_use_geometry",
                    "joint_task": "room co-use",
                    "constraint_kind": "interaction_distance",
                    "failure_condition": "objects do not share a task",
                },
            ],
            "cross_group_correspondences": [
                {
                    "target_ids": ["sofa", "television"],
                    "predicate": "directional_correspondence",
                    "dependency": "required",
                    "counterpart_mode": "dedicated",
                    "ordinary_mobility": "fixed",
                    "observation_goal": (
                        "inspect whether their facing directions permit "
                        "ordinary viewing"
                    ),
                    "discovery_id": "functional_correspondence_02",
                    "scope": "cross_group",
                },
                {
                    # A relation that did not satisfy the current contract is
                    # not forwarded merely because the objects co-occur.
                    "target_ids": ["lamp", "table"],
                    "predicate": "relative_use_geometry",
                    "joint_task": "",
                    "constraint_kind": "relative_alignment",
                    "failure_condition": "",
                },
            ],
        },
        "functional_measurement_bank": {
            "check_measurements": [
                {
                    "check_id": "functional_check_001",
                    "check_type": "clearance",
                    "target_ids": ["wardrobe"],
                    "measurement_extensions": {
                        "directional_clearance": {
                            "status": "available",
                            "usable_side_id": "local_neg_y",
                            "world_outward_direction_xy": [0.0, -1.0],
                            "frontage_origin_xy": [1.0, 2.0],
                            "corridor_depth_m": 1.2,
                            "corridor_half_width_m": 0.5,
                            "nearest_forward_obstacle_distance_m": 0.7,
                            "forward_intersections": [
                                {
                                    "object_id": "chair",
                                    "forward_near_distance_m": 0.7,
                                    "forward_far_distance_m": 1.1,
                                    "lateral_clearance_m": -0.2,
                                    "corridor_overlap_depth_m": 0.4,
                                    "corridor_overlap_width_m": 0.3,
                                    "corridor_width_overlap_fraction": 0.3,
                                    "corridor_overlap_area_proxy_m2": 0.12,
                                    "vertical_overlap_with_approach_m": 0.9,
                                    "vertical_relevant": True,
                                    "support_relation": "none",
                                    "thin_floor_layer": False,
                                    "ordinary_mobility": "movable_companion",
                                    "excluded_from_obstacle": False,
                                }
                            ],
                            "unavailable_reason": None,
                            "decision_authority": "none",
                        }
                    },
                }
            ]
        },
    }


def test_function_context_projects_only_clearance_and_gated_binary_relations(
) -> None:
    context = project_functional_spatial_context(
        _functional_report(),
        known_object_ids=[
            "wardrobe",
            "lamp",
            "chair",
            "table",
            "sofa",
            "television",
        ],
    )

    assert context == {
        "schema_version": "functional_spatial_context_v3",
        "context_role": "attention_only",
        "decision_authority": "none",
        "clearance_requirements": [
            {
                "object_id": "wardrobe",
                "observation_goal": "inspect ordinary opening clearance",
                "source_check_id": "approach_clearance_01",
                "ownership": "neutral_prerequisite",
                "measurement": {
                    "status": "available",
                    "usable_side_id": "local_neg_y",
                    "world_outward_direction_xy": [0.0, -1.0],
                    "frontage_origin_xy": [1.0, 2.0],
                    "corridor_depth_m": 1.2,
                    "corridor_half_width_m": 0.5,
                    "nearest_forward_obstacle_distance_m": 0.7,
                    "forward_intersections": [
                        {
                            "object_id": "chair",
                            "forward_near_distance_m": 0.7,
                            "forward_far_distance_m": 1.1,
                            "lateral_clearance_m": -0.2,
                            "corridor_overlap_depth_m": 0.4,
                            "corridor_overlap_width_m": 0.3,
                            "corridor_width_overlap_fraction": 0.3,
                            "corridor_overlap_area_proxy_m2": 0.12,
                            "vertical_overlap_with_approach_m": 0.9,
                            "vertical_relevant": True,
                            "support_relation": "none",
                            "thin_floor_layer": False,
                            "ordinary_mobility": "movable_companion",
                            "excluded_from_obstacle": False,
                        }
                    ],
                    "unavailable_reason": None,
                    "decision_authority": "none",
                },
            }
        ],
        "related_pairs": [
            {
                "object_ids": ["chair", "table"],
                "focal_id": "chair",
                "counterpart_id": "table",
                "predicate": "relative_use_geometry",
                "observation_goal": (
                    "inspect whether their relative distance permits "
                    "ordinary dining"
                ),
                    "source_relation_id": "functional_correspondence_01",
                    "scope": "within_group",
                    "dependency": "required",
                    "counterpart_mode": "dedicated",
                    "ordinary_mobility": "fixed",
                    "ownership": "function_owned",
                    "placement_role": "background_only",
            },
            {
                "object_ids": ["sofa", "television"],
                "focal_id": "sofa",
                "counterpart_id": "television",
                "predicate": "directional_correspondence",
                "observation_goal": (
                    "inspect whether their facing directions permit "
                    "ordinary viewing"
                ),
                    "source_relation_id": "functional_correspondence_02",
                    "scope": "cross_group",
                    "dependency": "required",
                    "counterpart_mode": "dedicated",
                    "ordinary_mobility": "fixed",
                    "ownership": "function_owned",
                    "placement_role": "background_only",
            },
        ],
        "projection_audit": _projection_audit(
            clearance=1,
            relations=2,
        ),
    }


def test_dense_context_is_bounded_audited_and_concurrently_deterministic() -> None:
    object_ids = [f"obj_{index:02d}" for index in range(40)]
    clearance_targets = []
    measurement_rows = []
    relations = []
    for index, object_id in enumerate(object_ids):
        clearance_targets.append(
            {
                "target_id": object_id,
                "need_clearance": True,
                "observation_goal": (
                    f"inspect ordinary access clearance for {object_id}"
                ),
                "discovery_id": f"clearance_{index:02d}",
            }
        )
        intersections = []
        for offset in range(1, 5):
            blocker = object_ids[(index + offset) % len(object_ids)]
            intersections.append(
                {
                    "object_id": blocker,
                    "forward_near_distance_m": 0.2 + 0.1 * offset,
                    "forward_far_distance_m": 0.5 + 0.1 * offset,
                    "lateral_clearance_m": -0.1,
                    "corridor_overlap_depth_m": 0.25,
                    "corridor_overlap_width_m": 0.2,
                    "corridor_width_overlap_fraction": 0.2,
                    "corridor_overlap_area_proxy_m2": 0.05,
                    "vertical_overlap_with_approach_m": 0.8,
                    "vertical_relevant": True,
                    "support_relation": "none",
                    "thin_floor_layer": False,
                    "ordinary_mobility": "fixed",
                    "excluded_from_obstacle": False,
                }
            )
        measurement_rows.append(
            {
                "check_id": f"functional_check_{index:03d}",
                "check_type": "clearance",
                "target_ids": [object_id],
                "measurement_extensions": {
                    "directional_clearance": {
                        "source": "deterministic_directional_clearance_v1",
                        "decision_authority": "none",
                        "measurement": {
                            "status": "available",
                            "usable_side_id": "local_neg_y",
                            "world_outward_direction_xy": [0.0, -1.0],
                            "frontage_origin_xy": [float(index), 1.0],
                            "corridor_depth_m": 1.2,
                            "corridor_half_width_m": 0.5,
                            "nearest_forward_obstacle_distance_m": 0.3,
                            "forward_intersections": intersections,
                            "unavailable_reason": None,
                            "decision_authority": "none",
                        },
                    }
                },
            }
        )
        relations.append(
            {
                "target_ids": [
                    object_id,
                    object_ids[(index + 1) % len(object_ids)],
                ],
                "predicate": "relative_use_geometry",
                "dependency": "required",
                "counterpart_mode": "shared",
                "ordinary_mobility": "fixed",
                "observation_goal": (
                    f"inspect ordinary use distance for relation {index:02d}"
                ),
                "discovery_id": f"relation_{index:02d}",
                "scope": "within_group",
            }
        )
    report = {
        "functional_discovery": {
            "schema_version": "functional_discovery_v5",
            "decision_authority": "none",
            "approach_clearance_targets": clearance_targets,
            "within_group_correspondences": relations,
            "cross_group_correspondences": [],
        },
        "functional_measurement_bank": {
            "schema_version": "functional_measurement_bank_v1",
            "check_measurements": measurement_rows,
        },
    }

    def project() -> dict:
        result = project_functional_spatial_context(
            report,
            known_object_ids=object_ids,
            max_context_chars=6500,
        )
        assert result is not None
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        projected = list(executor.map(lambda _index: project(), range(12)))

    assert all(item == projected[0] for item in projected[1:])
    context = projected[0]
    encoded = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    audit = context["projection_audit"]
    assert len(encoded) <= 6500
    assert audit["eligible_clearance_count"] == 40
    assert audit["eligible_relation_count"] == 40
    assert audit["emitted_clearance_count"] == len(
        context["clearance_requirements"]
    )
    assert audit["emitted_relation_count"] == len(
        context["related_pairs"]
    )
    assert (
        audit["eligible_clearance_count"]
        - audit["emitted_clearance_count"]
        == len(audit["omitted_clearance_object_ids"])
    )
    assert (
        audit["eligible_relation_count"]
        - audit["emitted_relation_count"]
        == len(audit["omitted_relation_source_ids"])
    )
    # Projection must not mutate or truncate the canonical Measurement Bank.
    assert len(report["functional_measurement_bank"]["check_measurements"]) == 40


def test_placement_discovery_receives_context_without_creating_a_check(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(image_path)
    model = _Model(
        {
            "considered_object_ids": ["wardrobe", "chair"],
            "candidates": [],
            "reason": "no independent semantic-location question",
        }
    )
    context = {
        "schema_version": "functional_spatial_context_v3",
        "context_role": "attention_only",
        "decision_authority": "none",
        "clearance_requirements": [
            {
                "object_id": "wardrobe",
                "observation_goal": "inspect ordinary opening clearance",
                "source_check_id": "approach_clearance_01",
                "ownership": "neutral_prerequisite",
                "measurement": None,
            }
        ],
        "related_pairs": [],
        "projection_audit": _projection_audit(
            clearance=1,
            relations=0,
        ),
    }

    result = discover_openai_compatible_placement_evidence(
        model=model,
        request={
            "metric": "semantic_placement_consistency",
            "scene_id": "scene",
            "scene_type": "bedroom",
            "global_image_path": str(image_path),
            "objects": [
                {"id": "wardrobe", "category": "wardrobe"},
                {"id": "chair", "category": "chair"},
            ],
            "functional_spatial_context": context,
        },
    )

    prompt_payload = model.calls[0]["messages"][1]["content"][0]["text"]
    assert '"functional_spatial_context"' in prompt_payload
    assert '"context_role":"attention_only"' in prompt_payload
    assert result["candidates"] == []


def test_invalid_optional_context_is_rejected_before_transport(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "global.png"
    Image.new("RGB", (16, 16), (100, 110, 120)).save(image_path)
    model = _Model({})

    with pytest.raises(ValueError, match="invalid object IDs"):
        discover_openai_compatible_placement_evidence(
            model=model,
            request={
                "metric": "semantic_placement_consistency",
                "global_image_path": str(image_path),
                "objects": [
                    {"id": "wardrobe", "category": "wardrobe"}
                ],
                "functional_spatial_context": {
                    "schema_version": "functional_spatial_context_v3",
                    "context_role": "attention_only",
                    "decision_authority": "none",
                    "clearance_requirements": [
                        {
                            "object_id": "unknown",
                            "observation_goal": "inspect opening clearance",
                            "source_check_id": "approach_clearance_01",
                            "ownership": "neutral_prerequisite",
                            "measurement": None,
                        }
                    ],
                    "related_pairs": [],
                    "projection_audit": _projection_audit(
                        clearance=1,
                        relations=0,
                    ),
                },
            },
        )

    assert model.calls == []


def test_context_only_relation_is_forwarded_as_placement_attention() -> None:
    report = deepcopy(_functional_report())
    report["functional_discovery"]["relation_admission_audit"] = {
        "context_only_relations": [
            {
                "proposal_ref": "relation_proposal_009",
                "target_ids": ["lamp", "table"],
                "focal_group_id": "group_a",
                "counterpart_group_id": "group_a",
                "predicate": "relative_use_geometry",
                "dependency": "contextual",
                "counterpart_mode": "shared",
                "ordinary_mobility": "portable_unrelated",
                "observation_goal": "inspect their contextual positioning",
            }
        ]
    }

    context = project_functional_spatial_context(
        report,
        known_object_ids=[
            "wardrobe",
            "lamp",
            "chair",
            "table",
            "sofa",
            "television",
        ],
    )

    relation = next(
        item
        for item in context["related_pairs"]
        if item["object_ids"] == ["lamp", "table"]
    )
    assert relation["ownership"] == "unowned_context"
    assert relation["placement_role"] == "candidate_attention"
    assert relation["scope"] == "within_group"
    assert relation["focal_id"] == "lamp"
    assert relation["counterpart_id"] == "table"


def test_reciprocal_relations_preserve_ordered_role_semantics() -> None:
    report = _functional_report()
    report["functional_discovery"]["within_group_correspondences"] = [
        {
            "target_ids": ["chair", "table"],
            "predicate": "relative_use_geometry",
            "dependency": "required",
            "counterpart_mode": "dedicated",
            "ordinary_mobility": "fixed",
            "observation_goal": "inspect chair use relative to table",
            "discovery_id": "relation_chair_to_table",
            "scope": "within_group",
        },
        {
            "target_ids": ["table", "chair"],
            "predicate": "relative_use_geometry",
            "dependency": "contextual",
            "counterpart_mode": "shared",
            "ordinary_mobility": "movable_companion",
            "observation_goal": "inspect table access relative to chair",
            "discovery_id": "relation_table_to_chair",
            "scope": "within_group",
        },
    ]
    report["functional_discovery"]["cross_group_correspondences"] = []

    context = project_functional_spatial_context(
        report,
        known_object_ids=["wardrobe", "lamp", "chair", "table"],
    )

    assert [
        (item["focal_id"], item["counterpart_id"])
        for item in context["related_pairs"]
    ] == [("chair", "table"), ("table", "chair")]
    assert context["related_pairs"][1]["ordinary_mobility"] == (
        "movable_companion"
    )


def test_bridge_compacts_to_structural_budget_with_explicit_audit() -> None:
    object_ids = [f"object_{index:02d}" for index in range(24)]
    approach = []
    checks = []
    relations = []
    for index in range(12):
        target_id = object_ids[index]
        approach.append(
            {
                "target_id": target_id,
                "need_clearance": True,
                "observation_goal": "inspect ordinary opening clearance",
                "discovery_id": f"approach_clearance_{index:02d}",
            }
        )
        intersections = []
        for offset in range(4):
            obstacle_id = object_ids[12 + ((index + offset) % 12)]
            intersections.append(
                {
                    "object_id": obstacle_id,
                    "forward_near_distance_m": 0.2 + offset,
                    "forward_far_distance_m": 0.8 + offset,
                    "lateral_clearance_m": -0.1,
                    "corridor_overlap_depth_m": 0.5,
                    "vertical_overlap_with_approach_m": 0.9,
                    "vertical_relevant": True,
                    "support_relation": "none",
                    "thin_floor_layer": False,
                    "ordinary_mobility": "fixed",
                    "excluded_from_obstacle": False,
                }
            )
        checks.append(
            {
                "check_id": f"functional_check_{index:03d}",
                "check_type": "clearance",
                "target_ids": [target_id],
                "measurement_extensions": {
                    "directional_clearance": {
                        "status": "available",
                        "usable_side_id": "local_neg_y",
                        "world_outward_direction_xy": [0.0, -1.0],
                        "frontage_origin_xy": [1.0, 2.0],
                        "corridor_depth_m": 1.2,
                        "corridor_half_width_m": 0.5,
                        "nearest_forward_obstacle_distance_m": 0.2,
                        "forward_intersections": intersections,
                        "unavailable_reason": None,
                        "decision_authority": "none",
                    }
                },
            }
        )
        relations.append(
            {
                "target_ids": [target_id, object_ids[12 + index]],
                "predicate": "relative_use_geometry",
                "dependency": "contextual",
                "counterpart_mode": "shared",
                "ordinary_mobility": "movable_companion",
                "observation_goal": "inspect contextual relative position",
                "proposal_ref": f"relation_proposal_{index:03d}",
                "focal_group_id": "group_a",
                "counterpart_group_id": "group_a",
            }
        )
    report = {
        "functional_discovery": {
            "decision_authority": "none",
            "approach_clearance_targets": approach,
            "within_group_correspondences": [],
            "cross_group_correspondences": [],
            "relation_admission_audit": {
                "context_only_relations": relations,
            },
        },
        "functional_measurement_bank": {
            "check_measurements": checks,
        },
    }

    context = project_functional_spatial_context(
        report,
        known_object_ids=object_ids,
        max_context_chars=4_000,
    )

    assert len(
        json.dumps(
            context,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) <= 4_000
    audit = context["projection_audit"]
    assert audit["eligible_clearance_count"] == 12
    assert audit["eligible_relation_count"] == 12
    assert (
        audit["reduced_measurement_object_ids"]
        or audit["omitted_measurement_object_ids"]
        or audit["omitted_relation_source_ids"]
    )
