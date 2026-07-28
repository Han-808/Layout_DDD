from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from benchmark.adapters.layout_json.converter import convert_layout_json_to_scene
from benchmark.evaluator.generic_validity.evaluator import evaluate_generic_validity
from benchmark.evaluator.object_alignment import evaluate_object_alignment
from benchmark.evaluator.object_mapping import evaluate_object_mapping, route_relationship_intents
from benchmark.reference_annotation import (
    ReferenceAnnotationError,
    annotation_scoring_gate,
    approve_reference_annotation,
    build_reference_annotation_draft,
    confirm_reference_annotation,
    is_official_scoreable,
    object_plan_from_reference_annotation,
    relationship_intents_from_reference_annotation,
    validate_reference_annotation,
)
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.object_normalization import normalize_object, rotation_matrix_from_euler
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene
from evaluate import run_evaluate
from scripts.run_scene_harness import run_scene_harness


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _scene(objects: list[dict], *, boundary: list[list[float]] | None = None, height: float = 2.8) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "generated_case_001",
        "request_id": "case_001",
        "scene_type": "room",
        "boundary": boundary or [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": height,
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


def _obj(object_id: str, category: str, description: str, *, center=None, size=None, rotation=None) -> dict:
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "size": size or [1.0, 1.0, 1.0],
        "center": list(center or [2.0, 1.5, 0.5]),
        "rotation": list(rotation or [0.0, 0.0, 0.0]),
        "metadata": {"interactive": False},
    }


def _annotation(
    objects: list[dict],
    *,
    inventory_policy: str = "open_world",
    status: str = "confirmed",
    room: dict | None = None,
    oor: list[dict] | None = None,
    source: str = "programmatic",
) -> dict:
    annotation = {
        "annotation_version": "reference_annotation_v1",
        "validation_status": status,
        "source": source,
        "request_id": "case_001",
        "scene_type": "room",
        "objects": objects,
        "oor_relations": oor or [],
        "oar_relations": [],
        "room_constraints": room or {"claim_state": "not_mentioned"},
    }
    if inventory_policy is not None:
        annotation["inventory_policy"] = inventory_policy
    return annotation


def _ref_object(object_id: str, category: str, *, count: int = 1, claim_state: str = "confirmed") -> dict:
    return {"id": object_id, "category": category, "count": count, "claim_state": claim_state}


def _alignment(annotation: dict, scene: dict) -> dict:
    mapping = evaluate_object_mapping(object_plan_from_reference_annotation(annotation), scene)
    return evaluate_object_alignment(annotation, scene, mapping)


# --------------------------------------------------------------------------- #
# 1. Converter draft is not used as the official reference automatically
# --------------------------------------------------------------------------- #
def test_converter_draft_is_not_official_reference(tmp_path) -> None:
    object_plan = {
        "request_id": "case_001",
        "scene_type": "room",
        "objects": [{"id": "bed", "category": "bed", "count": 1}],
        "relations": [],
    }
    draft = build_reference_annotation_draft(object_plan)

    assert draft["validation_status"] == "draft"
    assert is_official_scoreable(draft) is False

    report = run_evaluate(
        scene=_scene([_obj("bed", "bed", "bed")]),
        out=tmp_path / "report.json",
        reference_annotation=draft,
    )
    assert "object_mapping" not in report["reports"]
    assert "object_alignment" not in report["reports"]
    assert report["category_reports"]["l2_specification_fidelity"]["score"] is None


def test_model_assisted_reference_requires_human_approval_and_preserves_claim_states() -> None:
    draft = build_reference_annotation_draft(
        {
            "request_id": "case_001",
            "scene_type": "room",
            "objects": [
                {"id": "bed", "category": "bed", "count": 1},
                {"id": "desk", "category": "desk", "count": 1},
            ],
            "relations": [
                {"family": "oor", "subject_id": "bed", "type": "near", "object_id": "desk"}
            ],
        },
        source="model_assisted",
    )
    draft["objects"][0]["claim_state"] = "confirmed"

    assert annotation_scoring_gate(draft)["reason"] == "reference_annotation_not_confirmed"
    approved = approve_reference_annotation(
        draft,
        inventory_policy="open_world",
        reviewer="human_reviewer",
        reviewed_at="2026-07-16T17:30:00+08:00",
    )

    assert annotation_scoring_gate(approved)["official_scoreable"] is True
    assert approved["objects"][0]["claim_state"] == "confirmed"
    assert approved["objects"][1]["claim_state"] == "unresolved_annotation"
    assert approved["oor_relations"][0]["claim_state"] == "unresolved_annotation"
    assert approved["review"]["reviewer"] == "human_reviewer"

    confirmed_via_legacy_helper = confirm_reference_annotation(
        draft,
        inventory_policy="open_world",
        reviewer="human_reviewer",
    )
    assert confirmed_via_legacy_helper["objects"][0]["claim_state"] == "confirmed"
    assert confirmed_via_legacy_helper["objects"][1]["claim_state"] == "unresolved_annotation"
    assert confirmed_via_legacy_helper["oor_relations"][0]["claim_state"] == "unresolved_annotation"


def test_unconfirmed_reference_relations_do_not_leak_into_p0b_context(tmp_path: Path) -> None:
    draft = build_reference_annotation_draft(
        {
            "request_id": "case_001",
            "scene_type": "room",
            "objects": [{"id": "lamp", "category": "lamp", "count": 1}],
            "relations": [
                {
                    "family": "oar",
                    "subject_id": "lamp",
                    "type": "against",
                    "architectural_element": "wall",
                }
            ],
        }
    )
    calls: list[dict] = []

    class Judge:
        def adjudicate_p0b(self, request: dict) -> dict:
            calls.append(request)
            return {"verdict": "valid", "confidence": 0.9, "reason": "test"}

    run_evaluate(
        scene=_scene([_obj("lamp", "lamp", "floor lamp", center=[2.0, 1.5, 1.5])]),
        out=tmp_path / "draft_p0b.json",
        reference_annotation=draft,
        eval_generic_validity=True,
        vlm_judge=Judge(),
    )

    assert calls
    assert all(call["extracted_relationships"] == [] for call in calls)


# --------------------------------------------------------------------------- #
# 2. Incomplete reference annotation cannot enter official scoring
# --------------------------------------------------------------------------- #
def test_incomplete_annotation_excluded_from_official_scoring() -> None:
    annotation = _annotation(
        [_ref_object("bed", "bed")],
        inventory_policy=None,
        status="reference_annotation_incomplete",
    )
    result = _alignment(annotation, _scene([_obj("bed", "bed", "bed")]))

    assert result["official_scoreable"] is False
    assert result["reason"] == "reference_annotation_incomplete"
    assert result["score"] is None


def test_known_incomplete_flag_blocks_confirmation() -> None:
    annotation = _annotation([_ref_object("bed", "bed")])
    annotation["known_incomplete"] = True
    with pytest.raises(ReferenceAnnotationError):
        validate_reference_annotation(annotation)


def test_closed_world_rejects_unresolved_object_inventory() -> None:
    annotation = _annotation(
        [
            _ref_object("bed", "bed", claim_state="confirmed"),
            _ref_object("desk", "desk", claim_state="unresolved_annotation"),
        ],
        inventory_policy="closed_world",
    )
    with pytest.raises(ReferenceAnnotationError, match="every listed object claim"):
        validate_reference_annotation(annotation)


# --------------------------------------------------------------------------- #
# 3. Only confirmed claims are eligible
# --------------------------------------------------------------------------- #
def test_only_confirmed_claims_are_eligible() -> None:
    annotation = _annotation(
        [
            _ref_object("bed", "bed", claim_state="confirmed"),
            _ref_object("desk", "desk", claim_state="unresolved_annotation"),
        ]
    )
    result = _alignment(annotation, _scene([_obj("bed", "bed", "bed")]))

    assert result["presence_evidence"]["eligible_count"] == 1
    assert [obj["reference_object_id"] for obj in result["objects"]] == ["bed"]
    assert result["presence_evidence"]["resolved_count"] == 1


# --------------------------------------------------------------------------- #
# 4. Missing confirmed objects affect presence / count
# --------------------------------------------------------------------------- #
def test_missing_confirmed_objects_are_classified_without_scoring_in_p0a() -> None:
    annotation = _annotation([_ref_object("bed", "bed"), _ref_object("desk", "desk")])
    result = _alignment(annotation, _scene([_obj("bed", "bed", "bed")]))

    assert result["state_counts"]["missing"] == 1
    assert result["state_counts"]["resolved"] == 1
    assert result["presence_evidence"]["missing_count"] == 1
    assert result["metric_role"] == "alignment_only"
    assert result["score"] is None
    assert "count_fidelity" not in result


# --------------------------------------------------------------------------- #
# 5. Ambiguous / low-confidence excluded only from affected relation checks
# --------------------------------------------------------------------------- #
def test_ambiguous_identities_excluded_only_from_relation_checks() -> None:
    # Two identical sofas make the single confirmed "sofa" identity ambiguous,
    # while "table" resolves uniquely.
    annotation = _annotation([_ref_object("sofa", "sofa"), _ref_object("table", "table")])
    scene = _scene(
        [
            _obj("sofa_a", "sofa", "sofa", center=[1.0, 1.0, 0.5]),
            _obj("sofa_b", "sofa", "sofa", center=[3.0, 1.0, 0.5]),
            _obj("table_1", "table", "table", center=[2.0, 2.5, 0.5]),
        ]
    )
    mapping = evaluate_object_mapping(object_plan_from_reference_annotation(annotation), scene)
    alignment = evaluate_object_alignment(annotation, scene, mapping)

    # Alignment: sofa is ambiguous but still present (not missing); table resolves.
    assert alignment["state_counts"]["ambiguous"] == 1
    assert alignment["state_counts"]["resolved"] == 1
    assert alignment["state_counts"]["missing"] == 0
    assert alignment["presence_evidence"]["ambiguous_count"] == 1

    # Relation routing: a relation whose subject is ambiguous is excluded from OOR.
    intents = {
        "request_id": "case_001",
        "oor_relations": [{"family": "oor", "subject_id": "sofa", "type": "near", "object_id": "table"}],
        "oar_relations": [],
    }
    routed = route_relationship_intents(intents, mapping)
    assert routed["oor_relations"] == []
    assert routed["alignment"]["resolved_count"] == 0
    assert routed["alignment"]["unresolved_count"] == 1


def test_low_confidence_identities_excluded_only_from_relation_checks() -> None:
    # A generic generated category with a partial description match yields a
    # deterministic similarity in [0.3, 0.5): a low-confidence, unresolved slot.
    object_plan = {
        "request_id": "case_001",
        "scene_type": "room",
        "scene_description": "",
        "objects": [
            {
                "id": "seat",
                "category": "object",
                "description": "red upholstered chair",
                "count": 1,
                "metadata": {},
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }
    scene = _scene([_obj("g0", "furniture", "red chair")])
    mapping = evaluate_object_mapping(object_plan, scene)
    assert [entry["plan_slot_id"] for entry in mapping["low_confidence_mappings"]] == ["seat"]

    annotation = _annotation([_ref_object("seat", "object")])
    alignment = evaluate_object_alignment(annotation, scene, mapping)
    assert alignment["state_counts"]["low_confidence"] == 1
    assert alignment["state_counts"]["missing"] == 0
    assert alignment["presence_evidence"]["low_confidence_count"] == 1

    intents = {
        "request_id": "case_001",
        "oor_relations": [{"family": "oor", "subject_id": "seat", "type": "near", "object_id": "seat"}],
        "oar_relations": [],
    }
    routed = route_relationship_intents(intents, mapping)
    assert routed["oor_relations"] == []
    assert routed["alignment"]["unresolved_count"] == 1


# --------------------------------------------------------------------------- #
# 6 & 7. Extras affect fidelity under closed_world only
# --------------------------------------------------------------------------- #
def test_closed_world_extras_are_reported_for_later_fidelity() -> None:
    annotation = _annotation([_ref_object("bed", "bed")], inventory_policy="closed_world")
    scene = _scene([_obj("bed", "bed", "bed"), _obj("vase", "vase", "ceramic vase", center=[3.0, 2.0, 0.5])])
    result = _alignment(annotation, scene)

    assert result["state_counts"]["extra"] == 1
    assert result["inventory_evidence"] == {"inventory_policy": "closed_world", "extra_count": 1}
    assert result["score"] is None


def test_open_world_extras_are_reported_without_p0a_penalty() -> None:
    annotation = _annotation([_ref_object("bed", "bed")], inventory_policy="open_world")
    scene = _scene([_obj("bed", "bed", "bed"), _obj("vase", "vase", "ceramic vase", center=[3.0, 2.0, 0.5])])
    result = _alignment(annotation, scene)

    assert result["state_counts"]["extra"] == 1
    assert result["inventory_evidence"] == {"inventory_policy": "open_world", "extra_count": 1}
    assert result["score"] is None


# --------------------------------------------------------------------------- #
# 8 & 9. Room fidelity depends on confirmed room constraints
# --------------------------------------------------------------------------- #
def test_unspecified_room_permits_generator_chosen_geometry() -> None:
    annotation = _annotation([_ref_object("bed", "bed")], room={"claim_state": "not_mentioned"})
    scene = _scene([_obj("bed", "bed", "bed")], boundary=[[0, 0], [6, 0], [6, 4], [0, 4]], height=3.1)

    assert validate_generated_scene(scene) is scene
    result = _alignment(annotation, scene)
    assert "room_fidelity" not in result


def test_generated_scene_contract_accepts_group_oor_and_rich_oar_relations() -> None:
    scene = _scene(
        [
            _obj("left", "chair", "left chair"),
            _obj("middle", "table", "middle table"),
            _obj("right", "chair", "right chair"),
            _obj("north", "chair", "north chair"),
        ]
    )
    scene["oor_relations"] = [
        {
            "family": "oor",
            "relation_id": "between_1",
            "type": "between",
            "subject_id": "middle",
            "object_ids": ["left", "right"],
            "raw_relation": "the table is between the chairs",
        },
        {
            "family": "oor",
            "relation_id": "ordered_1",
            "type": "ordered",
            "object_ids": ["left", "middle", "right"],
            "direction": "left_to_right",
        },
        {
            "family": "oor",
            "relation_id": "around_1",
            "type": "around",
            "subject_ids": ["left", "right", "north"],
            "object_id": "middle",
        },
    ]
    scene["oar_relations"] = [
        {
            "family": "oar",
            "relation_id": "wall_1",
            "subject_id": "left",
            "type": "near_wall",
            "architectural_element": "west_wall",
            "wall": "west",
            "raw_relation": "the left chair is near the west wall",
        }
    ]

    assert validate_generated_scene(scene) is scene
    schema = json.loads((ROOT / "schemas" / "scene.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(scene)) == []


def test_specified_room_constraints_remain_frozen_but_are_not_scored_in_p0a() -> None:
    room = {"claim_state": "confirmed", "dimensions": {"width": 4.0, "depth": 3.0, "height": 2.8}}
    annotation = _annotation([_ref_object("bed", "bed")], room=room)

    validate_reference_annotation(annotation)
    alignment = _alignment(
        annotation,
        _scene([_obj("bed", "bed", "bed")], boundary=[[0, 0], [8, 0], [8, 6], [0, 6]], height=3.5),
    )
    assert alignment["score"] is None
    assert "room_fidelity" not in alignment


def test_reference_annotation_rejects_unknown_relation_objects() -> None:
    annotation = _annotation(
        [_ref_object("bed", "bed")],
        oor=[
            {
                "subject_id": "bed",
                "type": "near",
                "object_id": "missing",
                "claim_state": "confirmed",
            }
        ],
    )
    with pytest.raises(ReferenceAnnotationError, match="unknown object id 'missing'"):
        validate_reference_annotation(annotation)


def test_confirmation_does_not_confirm_relation_to_unconfirmed_object() -> None:
    draft = build_reference_annotation_draft(
        {
            "request_id": "case_001",
            "scene_type": "room",
            "objects": [
                {"id": "bed", "category": "bed", "count": 1},
                {"id": "desk", "category": "desk", "count": 1},
            ],
            "relations": [
                {"family": "oor", "subject_id": "bed", "type": "near", "object_id": "desk"}
            ],
        }
    )
    confirmed = confirm_reference_annotation(
        draft,
        inventory_policy="open_world",
        confirmed_object_ids={"bed"},
    )
    assert confirmed["oor_relations"][0]["claim_state"] == "unresolved_annotation"


def test_confirmed_reference_relations_project_without_converter_mapping() -> None:
    annotation = _annotation(
        [_ref_object("bed", "bed"), _ref_object("desk", "desk")],
        oor=[
            {
                "subject_id": "bed",
                "type": "near",
                "object_id": "desk",
                "claim_state": "confirmed",
            }
        ],
    )
    intents = relationship_intents_from_reference_annotation(annotation)
    assert intents["oor_relations"] == [
        {
            "relation_id": "oor_000",
            "relation_id_generated": True,
            "relation_id_provenance": "legacy_family_index",
            "family": "oor",
            "subject_id": "bed",
            "type": "near",
            "object_id": "desk",
            "source": "frozen_reference_annotation",
        }
    ]


def test_relation_ids_are_authored_preserved_and_duplicate_checked() -> None:
    draft = build_reference_annotation_draft(
        {
            "request_id": "case_001",
            "scene_type": "room",
            "objects": [
                {"id": "bed", "category": "bed", "count": 1},
                {"id": "desk", "category": "desk", "count": 1},
            ],
            "relations": [
                {"family": "oor", "subject_id": "bed", "type": "near", "object_id": "desk"},
                {"family": "oar", "subject_id": "bed", "type": "on_floor"},
            ],
        }
    )
    assert draft["oor_relations"][0]["relation_id"] == "oor_000"
    assert draft["oar_relations"][0]["relation_id"] == "oar_000"

    confirmed = confirm_reference_annotation(draft, inventory_policy="closed_world")
    intents = relationship_intents_from_reference_annotation(confirmed)
    assert intents["oor_relations"][0]["relation_id"] == "oor_000"
    assert intents["oar_relations"][0]["relation_id"] == "oar_000"

    duplicate = _annotation(
        [_ref_object("bed", "bed"), _ref_object("desk", "desk")],
        oor=[
            {"relation_id": "same", "subject_id": "bed", "type": "near", "object_id": "desk", "claim_state": "confirmed"},
            {"relation_id": "same", "subject_id": "desk", "type": "near", "object_id": "bed", "claim_state": "confirmed"},
        ],
    )
    with pytest.raises(ReferenceAnnotationError, match="duplicates relation id"):
        validate_reference_annotation(duplicate)


def test_blank_source_relation_id_cannot_override_authored_id() -> None:
    draft = build_reference_annotation_draft(
        {
            "request_id": "req_blank_relation_id",
            "scene_type": "room",
            "objects": [
                {"id": "bed", "category": "bed", "description": "bed", "count": 1},
                {"id": "desk", "category": "desk", "description": "desk", "count": 1},
            ],
            "relations": [
                {
                    "relation_id": "   ",
                    "family": "oor",
                    "subject_id": "bed",
                    "type": "near",
                    "object_id": "desk",
                }
            ],
        }
    )

    assert draft["oor_relations"][0]["relation_id"] == "oor_000"


def test_evaluation_api_preserves_on_top_relation_identity_and_reuses_support(tmp_path) -> None:
    scene = _scene(
        [
            _obj(
                "table_generated",
                "table",
                "wooden table",
                center=[2.0, 1.5, 0.375],
                size=[2.0, 1.5, 0.75],
            ),
            _obj(
                "clock_generated",
                "clock",
                "alarm clock",
                center=[2.0, 1.5, 0.85],
                size=[0.2, 0.2, 0.2],
            ),
        ]
    )
    annotation = _annotation(
        [
            {
                "id": "table_ref",
                "category": "table",
                "description": "wooden table",
                "count": 1,
                "claim_state": "confirmed",
            },
            {
                "id": "clock_ref",
                "category": "clock",
                "description": "alarm clock",
                "count": 1,
                "claim_state": "confirmed",
            },
        ],
        oor=[
            {
                "relation_id": "clock_on_table",
                "subject_id": "clock_ref",
                "type": "on_top_of",
                "object_id": "table_ref",
                "claim_state": "confirmed",
            }
        ],
    )

    report = run_evaluate(
        scene=scene,
        out=tmp_path / "on_top_report.json",
        reference_annotation=annotation,
        eval_oor=True,
        eval_generic_validity=True,
    )

    check = report["reports"]["oor"]["checks"][0]
    support = report["reports"]["generic_validity"]["metrics"]["support"]
    assert check["relation_id"] == "clock_on_table"
    assert check["subject_id"] == "clock_generated"
    assert check["object_id"] == "table_generated"
    assert check["route"] == "direct_valid"
    assert check["passed"] is True
    assert support["metric"] == "support"
    assert check["evidence"]["claimed_anchor_first_support_hit_count"] > 0


def test_reference_annotation_request_id_mismatch_is_excluded(tmp_path) -> None:
    annotation = _annotation([_ref_object("bed", "bed")])
    annotation["request_id"] = "another_case"
    report = run_evaluate(
        scene=_scene([_obj("bed", "bed", "bed")]),
        out=tmp_path / "report.json",
        reference_annotation=annotation,
    )
    assert "object_mapping" not in report["reports"]
    assert "object_alignment" not in report["reports"]
    assert report["category_reports"]["l2_specification_fidelity"]["score"] is None


def test_direct_evaluator_rejects_cross_request_prompt_and_plan_context(tmp_path) -> None:
    scene = _scene([_obj("bed", "bed", "bed")])
    with pytest.raises(ValueError, match="scene_request.request_id"):
        run_evaluate(
            scene=scene,
            out=tmp_path / "wrong_request.json",
            scene_request={"request_id": "other_case", "instruction": "Other prompt"},
        )

    wrong_plan = {
        "request_id": "other_case",
        "scene_type": "room",
        "scene_description": "Other plan",
        "objects": [],
        "global_constraints": [],
        "relations": [],
    }
    with pytest.raises(ValueError, match="object_plan.request_id"):
        run_evaluate(
            scene=scene,
            out=tmp_path / "wrong_plan.json",
            object_plan=wrong_plan,
        )


def test_confirmed_annotation_is_the_only_source_for_relation_routing(tmp_path) -> None:
    scene = _scene(
        [
            _obj("generated_bed", "bed", "bed", center=[1.0, 1.0, 0.5]),
            _obj("generated_desk", "desk", "desk", center=[2.0, 1.0, 0.5]),
        ]
    )
    annotation = _annotation(
        [_ref_object("bed", "bed"), _ref_object("desk", "desk")],
        oor=[
            {
                "subject_id": "bed",
                "type": "near",
                "object_id": "desk",
                "claim_state": "confirmed",
            }
        ],
    )
    report = run_evaluate(
        scene=scene,
        out=tmp_path / "confirmed.json",
        reference_annotation=annotation,
        eval_oor=True,
    )

    mapping = report["reports"]["object_mapping"]
    assert mapping["reference_source"] == "frozen_reference_annotation"
    assert mapping["official_scoreable"] is True
    assert report["reports"]["oor"]["status"] == "ok"
    assert report["reports"]["oor"]["num_checks_called"] == 1


def test_public_generator_structure_never_routes_or_scores_relations(tmp_path) -> None:
    scene = _scene(
        [
            _obj("generated_bed", "bed", "bed", center=[1.0, 1.0, 0.5]),
            _obj("generated_desk", "desk", "desk", center=[2.0, 1.0, 0.5]),
        ]
    )
    plan = {
        "request_id": "case_001",
        "scene_type": "room",
        "scene_description": "bed near desk",
        "prompt_granularity": "fine_grained",
        "explicit_claims": [],
        "objects": [
            {
                "id": "bed",
                "category": "bed",
                "description": "bed",
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            },
            {
                "id": "desk",
                "category": "desk",
                "description": "desk",
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            },
        ],
        "global_constraints": [],
        "relations": [
            {"family": "oor", "subject_id": "bed", "type": "near", "object_id": "desk"}
        ],
    }
    report = run_evaluate(
        scene=scene,
        out=tmp_path / "draft.json",
        object_plan=plan,
        eval_oor=True,
    )

    assert "object_mapping" not in report["reports"]
    assert "object_alignment" not in report["reports"]
    assert "oor" not in report["reports"]
    fidelity = report["reports"]["specification_fidelity"]
    assert fidelity["status"] == "not_evaluable"
    assert fidelity["reason"] == "missing_specification_contract"
    assert fidelity["activation_source"] == "benchmark_owned_specification_contract"
    assert fidelity["active_claim_families"] == []


def test_harness_keeps_reference_annotation_private_and_uses_public_room_fallback(tmp_path) -> None:
    out_dir = tmp_path / "frozen_case"
    annotation = _annotation([_ref_object("bed", "bed")])
    annotation["request_id"] = out_dir.name
    object_plan = {
        "request_id": out_dir.name,
        "scene_type": "room",
        "scene_description": "one bed",
        "prompt_granularity": "fine_grained",
        "explicit_claims": ["one bed"],
        "objects": [
            {
                "id": "bed",
                "role": "",
                "category": "bed",
                "description": "bed",
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {},
            }
        ],
        "global_constraints": [],
        "relations": [],
    }

    manifest = run_scene_harness(
        instruction="one bed",
        scene_type="room",
        out_dir=out_dir,
        object_plan=object_plan,
        reference_annotation=annotation,
    )

    generation_input = json.loads((out_dir / "generation_input.json").read_text(encoding="utf-8"))
    assert "reference_annotation" not in generation_input
    resolved_room = generation_input["scene_request"]["room"]
    assert resolved_room["dimensions"] == {"width": 7.0, "depth": 5.0, "height": 3.0}
    assert resolved_room["explicit_dimensions"] == {}
    assert set(resolved_room["dimension_provenance"].values()) == {"benchmark_fallback"}
    assert manifest["artifacts"]["reference_annotation"] == (out_dir / "reference_annotation.json").as_posix()


def test_harness_blocks_reference_annotation_leak_into_self_reflection(tmp_path) -> None:
    out_dir = tmp_path / "private_reflection_case"
    annotation = _annotation([_ref_object("bed", "bed")])
    annotation["request_id"] = out_dir.name
    with pytest.raises(ValueError, match="benchmark-private alignment evidence"):
        run_scene_harness(
            instruction="one bed",
            scene_type="room",
            out_dir=out_dir,
            object_plan={
                "request_id": out_dir.name,
                "scene_type": "room",
                "scene_description": "one bed",
                "prompt_granularity": "fine_grained",
                "explicit_claims": ["one bed"],
                "objects": [
                    {
                        "id": "bed",
                        "role": "",
                        "category": "bed",
                        "description": "bed",
                        "count": 1,
                        "placement_intent": {"absolute_relations": [], "relative_relations": []},
                        "metadata": {},
                    }
                ],
                "global_constraints": [],
                "relations": [],
            },
            reference_annotation=annotation,
            iteration_limit=1,
        )


# --------------------------------------------------------------------------- #
# 10. Absent canonical room fails without fallback
# --------------------------------------------------------------------------- #
def test_absent_canonical_room_fails_without_fallback() -> None:
    no_boundary = _scene([_obj("bed", "bed", "bed")])
    no_boundary.pop("boundary")
    with pytest.raises(ArtifactValidationError, match="boundary"):
        validate_generated_scene(no_boundary)

    no_height = _scene([_obj("bed", "bed", "bed")])
    no_height.pop("scene_height")
    with pytest.raises(ArtifactValidationError, match="scene_height"):
        validate_generated_scene(no_height)


# --------------------------------------------------------------------------- #
# 11. OOB objects pass schema and reach OOB evaluation
# --------------------------------------------------------------------------- #
def test_out_of_bounds_object_passes_schema_and_reaches_oob() -> None:
    scene = _scene([_obj("bed", "bed", "bed", center=[20.0, 20.0, 0.5])])

    assert validate_generated_scene(scene) is scene
    report = evaluate_generic_validity(scene)
    oob = report["metrics"]["oob"]
    # No judge is configured, so a flagged object is unresolved rather than a pass.
    assert oob["status"] == "requires_vlm"
    assert oob["score"] is None
    assert oob["candidate_oob_count"] >= 1
    record = oob["objects"][0]
    assert record["candidate_oob"] is True
    assert record["requires_vlm"] is True
    assert record["plane_flags"]["east_oob"] is True


# --------------------------------------------------------------------------- #
# 12 & 13. Exact OBB / rotation semantics
# --------------------------------------------------------------------------- #
def test_obb_corners_match_frozen_matrix_formula() -> None:
    rotation = [15.0, 30.0, 45.0]
    center = np.array([1.0, 2.0, 0.5])
    size = np.array([2.0, 1.0, 0.8])
    obj = normalize_object({"id": "o", "category": "box", "center": center.tolist(), "size": size.tolist(), "rotation": rotation})

    expected_R = rotation_matrix_from_euler(rotation)
    assert np.allclose(obj.R, expected_R)

    local_corner = np.array([size[0] / 2.0, size[1] / 2.0, size[2] / 2.0])
    expected_world = center + expected_R @ local_corner
    corners = center + (np.array([[1.0, 1.0, 1.0]]) * (size / 2.0)) @ expected_R.T
    assert np.allclose(corners[0], expected_world)


def test_local_obb_size_is_not_world_aabb_size() -> None:
    obj = normalize_object(
        {"id": "o", "category": "box", "center": [1.0, 1.0, 0.5], "size": [2.0, 1.0, 1.0], "rotation": [0.0, 0.0, 90.0]}
    )
    R = obj.R
    half = obj.half
    local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float) * half
    world = obj.center + local @ R.T
    aabb_width = float(world[:, 0].max() - world[:, 0].min())
    # A 90 deg yaw swaps footprint extents: world x-extent tracks local size_y (1.0), not size_x (2.0).
    assert abs(aabb_width - 1.0) < 1e-6
    assert abs(aabb_width - 2.0) > 0.5


# --------------------------------------------------------------------------- #
# 14. Geometry provenance is preserved
# --------------------------------------------------------------------------- #
def test_geometry_provenance_is_preserved() -> None:
    layout = _layout_json()
    generation_input = _generation_input()
    scene = convert_layout_json_to_scene(layout, generation_input)
    assert all(obj["geometry_provenance"] == "bbox_proxy" for obj in scene["objects"])

    normalized = normalize_scene(scene)
    assert normalized["objects"][0]["geometry_provenance"] == "bbox_proxy"

    invalid = _scene([_obj("bed", "bed", "bed")])
    invalid["objects"][0]["geometry_provenance"] = "hand_wave"
    with pytest.raises(ArtifactValidationError, match="geometry_provenance"):
        validate_generated_scene(invalid)


# --------------------------------------------------------------------------- #
# 15. No resolver or model fallback is called
# --------------------------------------------------------------------------- #
def test_no_resolver_or_model_fallback_in_alignment() -> None:
    for func in (evaluate_object_alignment, evaluate_object_mapping):
        params = set(inspect.signature(func).parameters)
        assert "resolver" not in params
        assert "model" not in params
        assert not any("resolver" in name for name in params)

    with pytest.raises(ModuleNotFoundError):
        import benchmark.evaluator.object_mapping_resolver  # noqa: F401

    result = _alignment(_annotation([_ref_object("bed", "bed")]), _scene([_obj("bed", "bed", "bed")]))
    assert "no model or resolver is called." in " ".join(result["notes"]).lower() or any(
        "no model" in note.lower() for note in result["notes"]
    )


# --------------------------------------------------------------------------- #
# Programmatic structural spec is preserved as a confirmable frozen annotation
# --------------------------------------------------------------------------- #
def test_programmatic_spec_can_be_frozen_as_confirmed_reference() -> None:
    object_plan = {
        "request_id": "case_001",
        "scene_type": "room",
        "objects": [{"id": "bed", "category": "bed", "count": 1}],
        "relations": [],
    }
    draft = build_reference_annotation_draft(object_plan, source="programmatic")
    confirmed = confirm_reference_annotation(draft, inventory_policy="closed_world")

    assert confirmed["validation_status"] == "confirmed"
    assert confirmed["inventory_policy"] == "closed_world"
    assert is_official_scoreable(confirmed) is True


def _layout_json() -> dict:
    return {
        "schema_version": "layout_json_v1",
        "scene_id": "layout_case",
        "scene_type": "bedroom",
        "coordinate_frame": {
            "origin": "room_min_corner_floor",
            "axes": "x_width_y_depth_z_up",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "room": {"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
        "objects": [
            {
                "id": "bed_1",
                "category": "bed",
                "description": "red bed",
                "center": [2.0, 1.5, 0.3],
                "size": [2.0, 1.6, 0.6],
                "rotation": [0.0, 0.0, 0.0],
            }
        ],
        "relationships": [],
    }


def _generation_input() -> dict:
    return {
        "request_id": "case_001",
        "scene_request": {
            "request_id": "case_001",
            "scene_type": "bedroom",
            "room": {"boundary": [[0, 0], [7, 0], [7, 5], [0, 5]], "height": 3.0},
        },
    }
