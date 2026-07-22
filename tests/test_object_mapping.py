from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmark.evaluator.object_mapping import evaluate_object_mapping, route_relationship_intents
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene, validate_object_plan


def test_object_mapping_is_global_semantic_and_does_not_require_shared_ids() -> None:
    plan = _plan(
        [
            _plan_object("reference_bed", "bed", "red upholstered bed"),
            _plan_object("reference_drawer", "drawer", "dark wood drawer"),
        ]
    )
    scene = _scene(
        [
            _scene_object("model_item_9", "drawer", "dark wood drawer", center=[7.0, 7.0, 0.4]),
            _scene_object("model_item_2", "bed", "red upholstered bed", center=[0.5, 0.5, 0.5]),
        ]
    )

    report = evaluate_object_mapping(plan, scene)

    assert report["status"] == "complete"
    assert report["placement_evidence_used"] is False
    assert report["resolved_mapping"] == {
        "reference_bed": ["model_item_2"],
        "reference_drawer": ["model_item_9"],
    }
    assert report["summary"]["object_count_exact"] is True
    assert report["summary"]["resolved_coverage"] == 1.0
    assert report["metric_role"] == "alignment_only"
    assert report["score"] is None
    assert report["penalty"] is None
    assert report["affects_benchmark_score"] is False


def test_object_mapping_expands_counts_and_reports_extra_generated_objects() -> None:
    plan = _plan([_plan_object("magazine", "magazine", "magazine", count=2)])
    scene = _scene(
        [
            _scene_object("mag_a", "magazine", "magazine"),
            _scene_object("mag_b", "magazine", "magazine"),
            _scene_object("extra_vase", "vase", "ceramic vase"),
        ]
    )

    report = evaluate_object_mapping(plan, scene)

    assert report["status"] == "partial"
    assert report["summary"]["reference_instance_count"] == 2
    assert report["summary"]["resolved_match_count"] == 2
    assert report["summary"]["ambiguous_match_count"] == 0
    assert report["summary"]["missing_reference_count"] == 0
    assert report["summary"]["extra_generated_count"] == 1
    assert report["unmatched_generated"][0]["generated_object_id"] == "extra_vase"


def test_object_mapping_keeps_indistinguishable_cross_object_matches_unresolved() -> None:
    plan = _plan(
        [
            _plan_object("left_lamp", "lamp", "small table lamp"),
            _plan_object("right_lamp", "lamp", "small table lamp"),
        ]
    )
    scene = _scene(
        [
            _scene_object("lamp_x", "lamp", "small table lamp"),
            _scene_object("lamp_y", "lamp", "small table lamp"),
        ]
    )

    report = evaluate_object_mapping(plan, scene)

    assert report["status"] == "partial"
    assert report["summary"]["ambiguous_match_count"] == 2
    assert report["summary"]["resolved_match_count"] == 0
    assert report["summary"]["unresolved_count"] == 2
    assert report["summary"]["mapping_coverage"] == 0.0
    assert report["summary"]["candidate_assignment_coverage"] == 1.0
    assert len(report["ambiguous_mappings"]) == 2
    assert all(item["status"] == "insufficient_alignment" for item in report["ambiguous_mappings"])
    assert all(item["score_effect"] == "none" for item in report["ambiguous_mappings"])
    # No model is ever consulted: ambiguous slots are left unresolved and never
    # entered into the resolved mapping used for downstream relation routing.
    assert report["resolved_mapping"] == {}
    assert {item["reason"] for item in report["unresolved_mappings"]} == {"ambiguous"}


def test_object_mapping_ignores_articles_and_quantity_tokens_in_descriptions() -> None:
    matching = evaluate_object_mapping(
        _plan([_plan_object("bed", "bed", "one red bed")]),
        _scene([_scene_object("generated_bed", "bed", "a red bed")]),
    )
    mismatch = evaluate_object_mapping(
        _plan([_plan_object("bed", "bed", "one red bed")]),
        _scene([_scene_object("generated_chair", "chair", "a blue chair")]),
    )

    assert matching["matches"][0]["score_components"]["description"] == 1.0
    assert mismatch["candidate_evidence"][0]["top_candidates"][0]["score_components"]["description"] == 0.0


def test_object_mapping_resolves_unique_exact_description_despite_category_drift() -> None:
    report = evaluate_object_mapping(
        _plan([_plan_object("reference", "object", "red upholstered seat")]),
        _scene([_scene_object("generated", "furniture", "red upholstered seat")]),
    )

    assert report["status"] == "complete"
    assert report["resolved_mapping"] == {"reference": ["generated"]}
    assert report["matches"][0]["score"] == 0.6
    assert report["matches"][0]["method"] == "description_only"
    assert report["low_confidence_mappings"] == []


def test_object_mapping_marks_partial_description_only_match_low_confidence() -> None:
    report = evaluate_object_mapping(
        _plan([_plan_object("reference", "object", "red upholstered seat")]),
        _scene([_scene_object("generated", "furniture", "red seat")]),
    )

    assert report["matches"] == []
    assert report["candidate_evidence"][0]["top_candidates"][0]["score"] == 0.48
    assert len(report["low_confidence_mappings"]) == 1
    trigger = report["low_confidence_mappings"][0]
    assert trigger["score_range"] == {"minimum_inclusive": 0.3, "maximum_exclusive": 0.5}
    assert trigger["status"] == "insufficient_alignment"
    assert trigger["score_effect"] == "none"
    assert report["ambiguous_mappings"] == []
    assert {item["reason"] for item in report["unresolved_mappings"]} == {"low_confidence"}
    assert report["summary"]["low_confidence_count"] == 1
    assert report["summary"]["missing_reference_count"] == 0
    assert report["missing_mappings"] == []


def test_object_mapping_leaves_candidates_below_point_three_unmatched() -> None:
    report = evaluate_object_mapping(
        _plan([_plan_object("reference", "bed", "red upholstered bed")]),
        _scene([_scene_object("generated", "chair", "blue wooden chair")]),
    )

    assert report["matches"] == []
    assert report["low_confidence_mappings"] == []
    assert report["ambiguous_mappings"] == []
    assert {item["reason"] for item in report["unresolved_mappings"]} == {"no_confident_candidate"}


def test_object_mapping_ignores_asset_and_retrieval_metadata() -> None:
    plan = _plan([_plan_object("bed_slot", "bed", "red bed")])
    generated = _scene_object("generated_1", "chair", "wood chair")
    generated["retrieval_category"] = "bed"
    generated["short_desc"] = "red bed"
    generated["asset_ref"]["category"] = "bed"
    generated["metadata"]["asset_binding"] = {
        "status": "selected",
        "object_slot_id": "bed_slot",
        "instance_index": 0,
    }
    generated["metadata"]["generator_description"] = "red bed"

    report = evaluate_object_mapping(plan, _scene([generated]))

    assert report["matches"] == []
    candidate = report["candidate_evidence"][0]["top_candidates"][0]
    assert candidate["score_components"] == {"category": 0.0, "description": 0.0}


def test_relationship_routing_uses_only_unique_resolved_identity_links() -> None:
    mapping_report = {
        "resolved_mapping": {
            "bed": ["generated_bed"],
            "lamp": ["generated_lamp_a", "generated_lamp_b"],
        }
    }
    intents = {
        "request_id": "mapping_case",
        "oor_relations": [
            {"family": "oor", "subject_id": "lamp", "type": "near", "object_id": "bed"},
            {"family": "oor", "subject_id": "bed", "type": "near", "object_id": "missing"},
        ],
        "oar_relations": [
            {"family": "oar", "subject_id": "bed", "type": "against_wall", "target": "right_wall"}
        ],
        "unsupported_relations": [],
    }

    routed = route_relationship_intents(intents, mapping_report)

    assert routed["oor_relations"] == []
    assert routed["oar_relations"][0]["subject_id"] == "generated_bed"
    assert routed["alignment"] == {
        "role": "routing_only",
        "score_effect": "none",
        "total_relation_count": 3,
        "routed_relation_count": 1,
        "unresolved_relation_count": 2,
        "eligible_count": 3,
        "resolved_count": 1,
        "unresolved_count": 2,
        "coverage": 1 / 3,
    }
    assert all(item["score_effect"] == "none" for item in routed["unresolved_relations"])
    assert all(item["status"] == "insufficient_alignment" for item in routed["unresolved_relations"])


def test_relationship_routing_expands_group_predicates_without_changing_semantics() -> None:
    mapping_report = {
        "resolved_mapping": {
            "chairs": ["chair_1", "chair_2", "chair_3", "chair_4"],
            "table": ["table_generated"],
            "left": ["left_generated"],
            "right": ["right_generated"],
            "middle": ["middle_generated"],
        }
    }
    intents = {
        "request_id": "group_case",
        "oor_relations": [
            {"family": "oor", "type": "around", "subject_ids": ["chairs"], "object_id": "table"},
            {"family": "oor", "type": "between", "subject_id": "middle", "object_ids": ["left", "right"]},
            {
                "family": "oor",
                "type": "ordered",
                "object_ids": ["left", "middle", "right"],
                "direction": "left_to_right",
            },
        ],
        "oar_relations": [],
    }

    routed = route_relationship_intents(intents, mapping_report)

    assert routed["unresolved_relations"] == []
    assert routed["oor_relations"][0]["subject_ids"] == ["chair_1", "chair_2", "chair_3", "chair_4"]
    assert routed["oor_relations"][0]["object_id"] == "table_generated"
    assert routed["oor_relations"][1]["object_ids"] == ["left_generated", "right_generated"]
    assert routed["oor_relations"][2]["object_ids"] == ["left_generated", "middle_generated", "right_generated"]


def test_relationship_routing_preserves_ids_across_filtered_claims() -> None:
    mapping_report = {
        "resolved_mapping": {
            "a": ["generated_a"],
            "b": ["generated_b"],
            "c": ["generated_c"],
        }
    }
    intents = {
        "oor_relations": [
            {"relation_id": "oor_first", "family": "oor", "subject_id": "a", "type": "near", "object_id": "b"},
            {"relation_id": "oor_filtered", "family": "oor", "subject_id": "a", "type": "near", "object_id": "missing"},
            {"relation_id": "oor_last", "family": "oor", "subject_id": "c", "type": "near", "object_id": "b"},
        ],
        "oar_relations": [],
    }

    routed = route_relationship_intents(intents, mapping_report)

    assert [item["relation_id"] for item in routed["oor_relations"]] == ["oor_first", "oor_last"]
    assert routed["unresolved_relations"][0]["relation_id"] == "oor_filtered"


def test_canonical_scene_gate_rejects_duplicate_ids_nonfinite_values_and_bad_references() -> None:
    duplicate = _scene([_scene_object("same", "chair", "chair"), _scene_object("same", "table", "table")])
    with pytest.raises(ArtifactValidationError, match="duplicates generated object id"):
        validate_generated_scene(duplicate)

    nonfinite = _scene([_scene_object("chair", "chair", "chair")])
    nonfinite["objects"][0]["center"][0] = float("nan")
    with pytest.raises(ArtifactValidationError, match="must be finite"):
        validate_generated_scene(nonfinite)

    bad_reference = _scene([_scene_object("chair", "chair", "chair")])
    bad_reference["relations"] = [{"subject_id": "chair", "type": "near", "object_id": "missing"}]
    with pytest.raises(ArtifactValidationError, match="unknown object id 'missing'"):
        validate_generated_scene(bad_reference)


def test_canonical_scene_gate_requires_valid_polygon_and_coordinate_contract() -> None:
    bow_tie = _scene([_scene_object("chair", "chair", "chair")])
    bow_tie["boundary"] = [[0, 0], [4, 4], [0, 4], [4, 0]]
    with pytest.raises(ArtifactValidationError, match="non-self-intersecting polygon"):
        validate_generated_scene(bow_tie)

    wrong_axis = _scene([_scene_object("chair", "chair", "chair")])
    wrong_axis["metadata"]["coordinate_frame"]["axes"] = "x_right_y_up_z_back"
    with pytest.raises(ArtifactValidationError, match="axes must be 'x_width_y_depth_z_up'"):
        validate_generated_scene(wrong_axis)

    shifted = _scene([_scene_object("chair", "chair", "chair")])
    shifted["boundary"] = [[1, 2], [5, 2], [5, 5], [1, 5]]
    with pytest.raises(ArtifactValidationError, match="minimum x=0 and y=0"):
        validate_generated_scene(shifted)


def test_canonical_o1_scene_does_not_require_asset_fields() -> None:
    obj = _scene_object("chair", "chair", "wooden chair")
    for key in ["jid", "asset_ref", "asset_proxy"]:
        obj.pop(key)

    scene = _scene([obj])

    assert validate_generated_scene(scene) is scene


def test_canonical_scene_requires_frozen_schema_version() -> None:
    missing = _scene([_scene_object("chair", "chair", "chair")])
    missing.pop("schema_version")
    with pytest.raises(ArtifactValidationError, match="schema_version must be 'canonical_scene_v1'"):
        validate_generated_scene(missing)

    unknown = _scene([_scene_object("chair", "chair", "chair")])
    unknown["schema_version"] = "canonical_scene_v2"
    with pytest.raises(ArtifactValidationError, match="schema_version must be 'canonical_scene_v1'"):
        validate_generated_scene(unknown)


def test_canonical_scene_requires_room_geometry() -> None:
    no_boundary = _scene([_scene_object("chair", "chair", "chair")])
    no_boundary.pop("boundary")
    with pytest.raises(ArtifactValidationError, match="boundary"):
        validate_generated_scene(no_boundary)

    no_height = _scene([_scene_object("chair", "chair", "chair")])
    no_height.pop("scene_height")
    with pytest.raises(ArtifactValidationError, match="scene_height"):
        validate_generated_scene(no_height)

    nonpositive_height = _scene([_scene_object("chair", "chair", "chair")])
    nonpositive_height["scene_height"] = 0.0
    with pytest.raises(ArtifactValidationError, match="scene_height must be positive"):
        validate_generated_scene(nonpositive_height)


@pytest.mark.parametrize("field", ["category", "center", "size", "rotation"])
def test_canonical_scene_requires_object_category_and_geometry(field: str) -> None:
    scene = _scene([_scene_object("chair", "chair", "wooden chair")])
    scene["objects"][0].pop(field)
    with pytest.raises(ArtifactValidationError, match=field):
        validate_generated_scene(scene)


def test_canonical_scene_allows_missing_description_asset_and_relations() -> None:
    obj = _scene_object("chair", "chair", "wooden chair")
    for key in ["description", "jid", "asset_ref", "asset_proxy"]:
        obj.pop(key)

    scene = _scene([obj])
    scene.pop("relations", None)
    scene.pop("oor_relations", None)
    scene.pop("oar_relations", None)

    assert validate_generated_scene(scene) is scene


def test_canonical_scene_rejects_infinity_non_positive_size_and_malformed_vectors() -> None:
    infinity = _scene([_scene_object("chair", "chair", "chair")])
    infinity["objects"][0]["center"][1] = float("inf")
    with pytest.raises(ArtifactValidationError, match="must be finite"):
        validate_generated_scene(infinity)

    nonpositive_size = _scene([_scene_object("chair", "chair", "chair")])
    nonpositive_size["objects"][0]["size"][2] = 0.0
    with pytest.raises(ArtifactValidationError, match="must be positive"):
        validate_generated_scene(nonpositive_size)

    malformed_vector = _scene([_scene_object("chair", "chair", "chair")])
    malformed_vector["objects"][0]["center"] = [1.0, 2.0]
    with pytest.raises(ArtifactValidationError, match="must be a 3-vector"):
        validate_generated_scene(malformed_vector)


def test_canonical_scene_enforces_coordinate_metadata() -> None:
    missing_frame = _scene([_scene_object("chair", "chair", "chair")])
    missing_frame["metadata"].pop("coordinate_frame")
    with pytest.raises(ArtifactValidationError, match="coordinate_frame"):
        validate_generated_scene(missing_frame)

    wrong_unit = _scene([_scene_object("chair", "chair", "chair")])
    wrong_unit["metadata"]["coordinate_frame"]["unit"] = "centimeter"
    with pytest.raises(ArtifactValidationError, match="unit must be 'meter'"):
        validate_generated_scene(wrong_unit)

    extra_field = _scene([_scene_object("chair", "chair", "chair")])
    extra_field["metadata"]["coordinate_frame"]["handedness"] = "right"
    with pytest.raises(ArtifactValidationError, match="unknown fields"):
        validate_generated_scene(extra_field)


def test_canonical_relations_require_canonical_field_names() -> None:
    scene = _scene(
        [
            _scene_object("chair", "chair", "chair"),
            _scene_object("table", "table", "table"),
        ]
    )
    scene["relations"] = [{"subject": "chair", "predicate": "near", "object": "table"}]
    with pytest.raises(ArtifactValidationError, match="forbidden keys"):
        validate_generated_scene(scene)

    conflicting = _scene(
        [
            _scene_object("chair", "chair", "chair"),
            _scene_object("table", "table", "table"),
        ]
    )
    conflicting["relations"] = [
        {
            "subject_id": "chair",
            "type": "near",
            "object_id": "table",
            "subject": "table",
        }
    ]
    with pytest.raises(ArtifactValidationError, match="forbidden keys"):
        validate_generated_scene(conflicting)


def test_object_mapping_report_carries_no_resolver_terminology() -> None:
    report = evaluate_object_mapping(
        _plan([_plan_object("bed", "bed", "red bed")]),
        _scene([_scene_object("generated_bed", "bed", "red bed")]),
    )

    assert report["mapping_policy"] == "deterministic_only"
    for dead_field in ["resolver_requests", "resolver_execution", "resolver_decisions", "vlm_policy"]:
        assert dead_field not in report
    for neutral_field in ["ambiguous_mappings", "low_confidence_mappings", "unresolved_mappings"]:
        assert neutral_field in report


def test_object_mapping_resolver_is_fully_removed() -> None:
    import benchmark.evaluator as evaluator_pkg

    assert not hasattr(evaluator_pkg, "build_openai_compatible_object_mapping_resolver")
    assert not hasattr(evaluator_pkg, "OpenAICompatibleObjectMappingResolver")
    with pytest.raises(ModuleNotFoundError):
        import benchmark.evaluator.object_mapping_resolver  # noqa: F401

    project_root = Path(__file__).resolve().parents[1]
    for relative in ["evaluate.py", "scripts/run_scene_harness.py"]:
        text = (project_root / relative).read_text(encoding="utf-8")
        assert "object-mapping-resolver" not in text
        assert "object_mapping_resolver" not in text
    assert not (project_root / "configs" / "models" / "object_mapping_resolver.example.json").exists()


def test_adapter_normalization_is_strict_and_does_not_invent_defaults() -> None:
    raw = _scene([_scene_object("chair", "chair", "chair")])
    raw["metadata"] = {}
    raw["objects"][0].pop("rotation")
    raw["objects"][0].pop("jid")
    raw["objects"][0].pop("asset_ref")
    raw["objects"][0].pop("asset_proxy")

    with pytest.raises(ArtifactValidationError, match="coordinate_frame"):
        normalize_scene(raw)

    raw["metadata"] = _scene([])["metadata"]
    with pytest.raises(ArtifactValidationError, match="rotation"):
        normalize_scene(raw)

    with pytest.raises(ArtifactValidationError, match="objects must be a JSON list"):
        normalize_scene({**raw, "objects": None, "assets": []})


def test_adapter_normalization_does_not_migrate_architecture_relations() -> None:
    raw = _scene([_scene_object("bed", "bed", "bed")])
    raw["relations"] = [{"subject_id": "bed", "type": "near", "object_id": "right_wall"}]

    with pytest.raises(ArtifactValidationError, match="unknown object id 'right_wall'"):
        normalize_scene(raw)


def test_object_plan_gate_rejects_duplicate_ids_and_invalid_relation_references() -> None:
    duplicate = _plan([_plan_object("chair", "chair", "chair"), _plan_object("chair", "chair", "chair")])
    with pytest.raises(ArtifactValidationError, match="duplicates object_plan object id"):
        validate_object_plan(duplicate)

    bad_relation = _plan([_plan_object("chair", "chair", "chair")])
    bad_relation["relations"] = [
        {"family": "oor", "subject_id": "chair", "type": "near", "object_id": "missing"}
    ]
    with pytest.raises(ArtifactValidationError, match="unknown object id 'missing'"):
        validate_object_plan(bad_relation)


def _plan(objects: list[dict]) -> dict:
    return {
        "request_id": "mapping_case",
        "scene_type": "room",
        "scene_description": "mapping fixture",
        "prompt_granularity": "fine_grained",
        "explicit_claims": [],
        "objects": objects,
        "global_constraints": [],
        "relations": [],
    }


def _plan_object(object_id: str, category: str, description: str, *, count: int = 1) -> dict:
    return {
        "id": object_id,
        "role": "",
        "category": category,
        "description": description,
        "count": count,
        "placement_intent": {"absolute_relations": [], "relative_relations": []},
        "metadata": {},
    }


def _scene(objects: list[dict]) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "generated_mapping_case",
        "request_id": "mapping_case",
        "scene_type": "room",
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 2.8,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _scene_object(
    object_id: str,
    category: str,
    description: str,
    *,
    center: list[float] | None = None,
) -> dict:
    return {
        "id": object_id,
        "jid": f"proxy:{object_id}",
        "category": category,
        "description": description,
        "size": [1.0, 1.0, 1.0],
        "center": list(center or [2.0, 2.0, 0.5]),
        "rotation": [0.0, 0.0, 0.0],
        "asset_ref": {"source_db": "proxy", "asset_key": f"proxy:{object_id}"},
        "asset_proxy": {"type": "obb", "bbox_center_local": [0.0, 0.0, 0.0], "bbox_size": [1.0, 1.0, 1.0]},
        "metadata": {"interactive": False},
    }
