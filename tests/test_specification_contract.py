from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.evaluator import (
    CANONICAL_PROFILE_VERSION,
    LEGACY_PROFILE_VERSION,
    build_specification_fidelity_report,
    compile_specification_evaluation_plan,
    resolve_evaluation_profile,
    specification_activation_mode,
    specification_contract_from_reference_annotation,
    validate_specification_contract,
)
from benchmark.evaluator.specification_fidelity.contract import (
    FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES,
    SpecificationContractError,
    validate_frozen_cal_dataset2_v0_specification_contract,
)
from benchmark.evaluator.scene_quality import validate_authorized_deviations
from benchmark.evaluator.scene_quality.interfaces import SCENE_QUALITY_INTERFACE_METRICS
from evaluate import run_evaluate


CONTRACT_VERSION = "specification_contract_v1"


def _contract(claims: dict, *, source: str = "benchmark_annotation", frozen: bool = True) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "source": source,
        "frozen": frozen,
        "claims": claims,
    }


def _claim(claim_id: str, family: str, **extra) -> dict:
    return {"claim_id": claim_id, "claim_family": family, **extra}


# --- Compiler activation ------------------------------------------------------


def test_coarse_prompt_with_oor_activates_oor() -> None:
    contract = _contract({"oor": [_claim("oor_001", "oor", target_ids=["a", "b"])]})
    plan = compile_specification_evaluation_plan(contract, "coarse_grained")
    assert "oor" in plan["active_claim_families"]
    assert plan["modules"]["oor"]["active"] is True
    assert plan["prompt_granularity"] == "coarse_grained"
    assert plan["prompt_granularity_role"] == "metadata_and_reporting_slice"


def test_fine_prompt_with_required_area_component_activates_functional_semantics() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "rfa_001",
                    "functional_semantic_fidelity",
                    component="required_functional_areas",
                )
            ]
        }
    )
    plan = compile_specification_evaluation_plan(contract, "fine_grained")
    assert "functional_semantic_fidelity" in plan["active_claim_families"]
    module = plan["modules"]["functional_semantic_fidelity"]
    assert module["active"] is True
    assert module["implemented"] is True
    assert module["components"] == ["required_functional_areas"]
    assert module["local_functionality_claim_ids"] == []
    assert module["required_area_local_fallback_claim_ids"] == ["rfa_001"]


def test_canonical_local_functionality_claim_activates_conditional_local() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "local_001",
                    "functional_semantic_fidelity",
                    component="local_functionality",
                    target_ids=["desk", "chair"],
                )
            ]
        }
    )
    plan = compile_specification_evaluation_plan(contract, "coarse_grained")
    module = plan["modules"]["functional_semantic_fidelity"]
    assert module["components"] == ["local_functionality"]
    assert module["local_functionality_claim_ids"] == ["local_001"]
    assert module["local_evidence_condition"] == "prompt_specified_local_functionality"


def test_retired_families_are_rejected_by_canonical_runtime_and_schema() -> None:
    contract = _contract(
        {"object_count": [_claim("count::a", "object_count", target_ids=["a"])]}
    )
    with pytest.raises(SpecificationContractError, match="unknown families"):
        validate_specification_contract(contract)
    with pytest.raises(SpecificationContractError, match="unknown families"):
        compile_specification_evaluation_plan(contract, "coarse_grained")

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/specification_contract.schema.json").read_text(
            encoding="utf-8"
        )
    )
    errors = list(Draft202012Validator(schema).iter_errors(contract))
    assert errors


def test_frozen_cal_dataset2_v0_has_explicit_read_only_compatibility_boundary() -> None:
    claims = {family: [] for family in FROZEN_CAL_DATASET2_V0_CLAIM_FAMILIES}
    claims["object_count"] = [
        _claim("count::a", "object_count", target_ids=["a"])
    ]
    contract = _contract(claims)

    assert (
        validate_frozen_cal_dataset2_v0_specification_contract(
            contract,
            valid_object_ids={"a"},
        )
        is contract
    )
    with pytest.raises(SpecificationContractError, match="unknown families"):
        validate_specification_contract(contract)


def test_canonical_plan_and_report_do_not_emit_retired_compatibility_fields() -> None:
    contract = _contract(
        {"oor": [_claim("oor_001", "oor", target_ids=["a", "b"])]}
    )
    plan = compile_specification_evaluation_plan(contract, "coarse_grained")
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="coarse_grained",
        object_alignment_report={"official_scoreable": True},
        legacy_category_alias="prompt_fidelity",
    )
    forbidden = {
        "accepted_non_scoring_claim_families",
        "ignored_non_scoring_claims",
        "ignored_legacy_inputs",
        "legacy_category_alias",
    }
    assert forbidden.isdisjoint(plan)
    assert forbidden.isdisjoint(report)


def test_optional_claim_is_not_activated_or_scored() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "optional_room",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                    required=False,
                )
            ]
        }
    )
    plan = compile_specification_evaluation_plan(contract, "coarse_grained")
    assert plan["active_claim_families"] == []
    assert plan["ignored_optional_claims"] == {
        "functional_semantic_fidelity": ["optional_room"]
    }
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="coarse_grained",
    )
    assert report["status"] == "not_applicable"
    assert report["score"] is None
    assert report["coverage"]["eligible_claim_count"] == 0


def test_mixed_contract_activates_each_present_family() -> None:
    contract = _contract(
        {
            "oor": [_claim("oor_001", "oor", target_ids=["a", "b"])],
            "oar": [_claim("oar_001", "oar", subject_id="a", target_ids=["a"])],
            "functional_semantic_fidelity": [
                _claim(
                    "rst_001",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                ),
                _claim(
                    "rfa_001",
                    "functional_semantic_fidelity",
                    component="required_functional_areas",
                ),
            ],
        }
    )
    plan = compile_specification_evaluation_plan(contract, "coarse_grained")
    assert set(plan["active_claim_families"]) == {
        "oor",
        "oar",
        "functional_semantic_fidelity",
    }
    assert set(plan["modules"]) == {
        "oor",
        "oar",
        "functional_semantic_fidelity",
    }


def test_empty_family_is_not_applicable() -> None:
    contract = _contract({"oor": [_claim("oor_001", "oor", target_ids=["a", "b"])]})
    report = build_specification_fidelity_report(contract=contract, prompt_granularity="fine_grained")
    assert report["claim_family_reports"]["oar"]["status"] == "not_applicable"
    assert report["claim_family_reports"]["oar"]["score"] is None


def test_prompt_granularity_does_not_suppress_frozen_claims() -> None:
    contract = _contract(
        {
            "oor": [_claim("oor_001", "oor", target_ids=["a", "b"])],
            "functional_semantic_fidelity": [
                _claim(
                    "rst_001",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                )
            ],
        }
    )
    fine = compile_specification_evaluation_plan(contract, "fine_grained")
    coarse = compile_specification_evaluation_plan(contract, "coarse_grained")
    assert fine["active_claim_families"] == coarse["active_claim_families"]


# --- Report semantics ---------------------------------------------------------


def test_unexecuted_functional_runtime_does_not_become_zero_or_valid() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "rst_001",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                )
            ]
        }
    )
    report = build_specification_fidelity_report(contract=contract, prompt_granularity="coarse_grained")
    family = report["claim_family_reports"]["functional_semantic_fidelity"]
    assert family["status"] == "incomplete"
    assert family["score"] is None
    assert family["resolved_claim_count"] == 0
    assert family["claims"][0]["resolution"] == "unresolved"
    assert family["claims"][0]["reason"] == "evaluator_not_executed"
    assert family["claims"][0]["component"] == "room_scene_type"
    assert "legacy_claim_family" not in family["claims"][0]
    assert report["score"] is None
    assert report["score_role"].startswith(
        "equal_macro_average_across_active_metric_families"
    )
    assert report["coverage"]["complete"] is False


def test_missing_contract_official_is_not_evaluable() -> None:
    report = build_specification_fidelity_report(
        contract=None, prompt_granularity="fine_grained", official=True
    )
    assert report["status"] == "not_evaluable"
    assert report["reason"] == "missing_specification_contract"
    assert report["score"] is None


def test_missing_contract_diagnostic_is_explicit() -> None:
    report = build_specification_fidelity_report(
        contract=None,
        prompt_granularity="fine_grained",
        activation_mode="prompt_granularity_gate",
        official=False,
    )
    assert report["status"] == "not_applicable"
    assert report["reason"] == "missing_specification_contract"
    assert report["score"] is None


def test_case_level_scalar_not_comparable_note() -> None:
    contract = _contract({"oor": [_claim("oor_001", "oor", target_ids=["a", "b"])]})
    report = build_specification_fidelity_report(contract=contract, prompt_granularity="fine_grained")
    assert report["comparability"]["case_level_scalar_is_comparable"] is False


def test_functional_runtime_report_contributes_to_complete_canonical_score() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "room_001",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                ),
                _claim(
                    "area_001",
                    "functional_semantic_fidelity",
                    component="required_functional_areas",
                ),
            ]
        }
    )
    functional_report = {
        "metrics": {
            "functional_semantic_fidelity": {
                "checks": [
                    {"claim_id": "room_001", "status": "checked", "score": 1.0},
                    {"claim_id": "area_001", "status": "checked", "score": 0.0},
                ]
            }
        }
    }
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="coarse_grained",
        functional_semantic_report=functional_report,
    )
    assert report["status"] == "evaluated"
    assert report["score"] == 0.5
    assert report["coverage"]["complete"] is True
    family = report["claim_family_reports"]["functional_semantic_fidelity"]
    assert family["score"] == 0.5
    assert family["resolved_claim_count"] == 2


def test_one_unresolved_canonical_claim_suppresses_total_score() -> None:
    contract = _contract(
        {
            "functional_semantic_fidelity": [
                _claim(
                    "room_001",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                ),
                _claim(
                    "area_001",
                    "functional_semantic_fidelity",
                    component="required_functional_areas",
                ),
            ]
        }
    )
    functional_report = {
        "metrics": {
            "functional_semantic_fidelity": {
                "checks": [
                    {"claim_id": "room_001", "status": "checked", "score": 1.0},
                    {
                        "claim_id": "area_001",
                        "status": "requires_vlm",
                        "score": None,
                        "reason": "insufficient_evidence",
                    },
                ]
            }
        }
    }
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="coarse_grained",
        functional_semantic_report=functional_report,
    )
    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["partial_score"] == 1.0


def test_l2_uses_family_macro_not_claim_micro_aggregation() -> None:
    contract = _contract(
        {
            "oor": [
                _claim("oor_1", "oor", relation_id="oor_1"),
                _claim("oor_2", "oor", relation_id="oor_2"),
            ],
            "functional_semantic_fidelity": [
                _claim(
                    "room_1",
                    "functional_semantic_fidelity",
                    component="room_scene_type",
                )
            ],
        }
    )
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="coarse_grained",
        oor_report={
            "checks": [
                {"relation_id": "oor_1", "status": "checked", "score": 1.0},
                {"relation_id": "oor_2", "status": "checked", "score": 0.0},
            ]
        },
        functional_semantic_report={
            "metrics": {
                "functional_semantic_fidelity": {
                    "checks": [
                        {"claim_id": "room_1", "status": "checked", "score": 1.0}
                    ]
                }
            }
        },
    )
    assert report["claim_family_reports"]["oor"]["score"] == 0.5
    assert (
        report["claim_family_reports"]["functional_semantic_fidelity"]["score"]
        == 1.0
    )
    assert report["active_family_signature"] == [
        "oor",
        "functional_semantic_fidelity",
    ]
    assert report["score"] == 0.75
    assert report["resolved_claim_micro_average_diagnostic"] == pytest.approx(
        2.0 / 3.0
    )


def test_relation_invalid_input_is_failure_not_scene_score_zero() -> None:
    contract = _contract(
        {"oor": [_claim("oor_1", "oor", relation_id="oor_1")]}
    )
    report = build_specification_fidelity_report(
        contract=contract,
        prompt_granularity="fine_grained",
        oor_report={
            "checks": [
                {
                    "relation_id": "oor_1",
                    "status": "invalid_input",
                    "score": 0.0,
                }
            ]
        },
    )
    assert report["status"] == "incomplete"
    assert report["score"] is None
    check = report["claim_family_reports"]["oor"]["claims"][0]
    assert check["resolution"] == "failed"
    assert check["reason"] == "module_invalid_input"


# --- Contract validation ------------------------------------------------------


def test_duplicate_claim_ids_fail_validation() -> None:
    contract = _contract(
        {
            "oor": [_claim("dup", "oor", target_ids=["a", "b"])],
            "oar": [_claim("dup", "oar", subject_id="a", target_ids=["a"])],
        }
    )
    with pytest.raises(SpecificationContractError, match="unique"):
        validate_specification_contract(contract)


def test_untrusted_or_unfrozen_contract_rejected_for_official() -> None:
    with pytest.raises(SpecificationContractError, match="benchmark-owned"):
        validate_specification_contract(
            _contract({"oor": [_claim("oor_001", "oor")]}, source="programmatic"),
            require_trusted=True,
        )
    with pytest.raises(SpecificationContractError, match="frozen"):
        validate_specification_contract(
            _contract({"oor": [_claim("oor_001", "oor")]}, frozen=False),
            require_frozen=True,
        )
    with pytest.raises(SpecificationContractError, match="benchmark-owned"):
        build_specification_fidelity_report(
            contract=_contract(
                {"oor": [_claim("oor_001", "oor")]},
                source="programmatic",
            ),
            prompt_granularity="fine_grained",
            official=True,
        )
    with pytest.raises(SpecificationContractError, match="frozen"):
        build_specification_fidelity_report(
            contract=_contract(
                {"oor": [_claim("oor_001", "oor")]},
                frozen=False,
            ),
            prompt_granularity="fine_grained",
            official=True,
        )


def test_target_ids_must_exist_when_object_ids_supplied() -> None:
    contract = _contract({"oar": [_claim("oar_001", "oar", subject_id="ghost", target_ids=["ghost"])]})
    with pytest.raises(SpecificationContractError, match="unknown object id"):
        validate_specification_contract(contract, valid_object_ids={"a", "b"})
    with pytest.raises(SpecificationContractError, match="unknown object id"):
        validate_specification_contract(
            _contract(
                {
                    "functional_semantic_fidelity": [
                        _claim(
                            "local",
                            "functional_semantic_fidelity",
                            component="local_functionality",
                            object_ids=["ghost"],
                        )
                    ]
                }
            ),
            valid_object_ids={"a", "b"},
        )


def test_canonical_functional_semantic_claim_requires_known_component() -> None:
    with pytest.raises(SpecificationContractError, match="component"):
        validate_specification_contract(
            _contract(
                {
                    "functional_semantic_fidelity": [
                        _claim("fs_001", "functional_semantic_fidelity")
                    ]
                }
            )
        )


def test_activation_mode_from_profile_version() -> None:
    assert specification_activation_mode(LEGACY_PROFILE_VERSION) == "prompt_granularity_gate"
    assert specification_activation_mode(CANONICAL_PROFILE_VERSION) == "specification_contract"
    assert specification_activation_mode(None) == "specification_contract"


# --- Reference-annotation compiler --------------------------------------------


def _scene() -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "spec_scene",
        "request_id": "spec_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.9,
        "objects": [
            {
                "id": "bed",
                "category": "bed",
                "description": "blue bed",
                "desc": "blue bed",
                "size": [2.0, 1.6, 0.6],
                "center": [2.5, 2.5, 0.3],
                "rotation": [0, 0, 0],
                "asset_ref": {"source_db": "test", "asset_key": "bed_asset"},
                "asset_proxy": {"type": "obb", "bbox_center_local": [0, 0, 0], "bbox_size": [2.0, 1.6, 0.6]},
                "metadata": {"interactive": False},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _confirmed_annotation() -> dict:
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "inventory_policy": "open_world",
        "request_id": "spec_request",
        "scene_type": "bedroom",
        "objects": [
            {"id": "bed", "category": "bed", "description": "blue bed", "count": 1, "claim_state": "confirmed"}
        ],
        "oor_relations": [],
        "oar_relations": [
            {
                "subject_id": "bed",
                "type": "room_center",
                "architectural_element": "center_region",
                "claim_state": "confirmed",
            }
        ],
        "room_constraints": {"claim_state": "not_mentioned"},
    }


def test_compiler_maps_confirmed_annotation_without_new_truth() -> None:
    contract = specification_contract_from_reference_annotation(_confirmed_annotation())
    assert contract["source"] == "benchmark_annotation"
    assert contract["frozen"] is True
    assert "object_presence" not in contract["claims"]
    assert "object_count" not in contract["claims"]
    assert "explicit_attributes" not in contract["claims"]
    assert len(contract["claims"]["oar"]) == 1
    # High-level semantic truth is not fabricated from the current annotation schema.
    assert contract["claims"]["functional_semantic_fidelity"] == []
    assert json.loads(json.dumps(contract)) == contract


# --- run_evaluate integration -------------------------------------------------


def _run(profile_version: str | None, tmp_path: Path, name: str) -> dict:
    def judge(request: dict) -> dict:
        return {"score": 0.6}

    profile = resolve_evaluation_profile()
    if profile_version is not None:
        profile["profile_version"] = profile_version
    return run_evaluate(
        scene=_scene(),
        out=tmp_path / name,
        eval_oar=True,
        eval_generic_validity=True,
        scene_request={
            "request_id": "spec_request",
            "instruction": "Place a blue bed in the center.",
            "prompt_granularity": "fine_grained",
        },
        reference_annotation=_confirmed_annotation(),
        render_evidence=["standardized_top.png", "standardized_perspective.png"],
        vlm_judge=judge,
        evaluation_profile=profile,
    )


def test_run_evaluate_reuses_oar_and_preserves_scores(tmp_path: Path) -> None:
    legacy = _run(None, tmp_path, "legacy.json")
    v2 = _run(CANONICAL_PROFILE_VERSION, tmp_path, "v2.json")

    # OOR/OAR evaluator outputs are reused, not re-run or re-scored.
    assert "oar" in legacy["reports"]
    spec = legacy["reports"]["specification_fidelity"]
    assert spec["score"] == 1.0
    assert spec["claim_family_reports"]["oar"]["module"] == "oar"
    assert spec["claim_family_reports"]["oar"]["resolved_claim_count"] == 1
    assert "object_presence" not in spec["claim_family_reports"]

    # Explicit legacy-profile input and canonical-v2 input preserve numeric
    # aggregation and category reports for this fully resolved case.
    assert legacy["benchmark_score"] == v2["benchmark_score"]
    assert legacy["category_reports"] == v2["category_reports"]

    # Canonical routing is contract-driven regardless of the legacy input label.
    for output in (legacy, v2):
        activation = output["evaluation_config"]["specification_activation"]
        assert activation["source"] == "benchmark_owned_specification_contract"
        assert activation["prompt_granularity_controls_activation"] is False


def test_missing_contract_emits_explicit_canonical_coverage(tmp_path: Path) -> None:
    def judge(request: dict) -> dict:
        return {"score": 0.6}

    # The canonical default always emits an explicit L2 contract-coverage report.
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "legacy_coarse.json",
        eval_generic_validity=True,
        scene_request={
            "request_id": "spec_request",
            "instruction": "A bedroom.",
            "prompt_granularity": "coarse_grained",
        },
        render_evidence=["standardized_top.png"],
        vlm_judge=judge,
    )
    assert report["reports"]["specification_fidelity"]["status"] == "not_evaluable"
    assert report["reports"]["specification_fidelity"]["reason"] == "missing_specification_contract"


# --- Authorized deviation linkage ---------------------------------------------


def test_authorized_deviation_may_reference_source_claim_id() -> None:
    normalized = validate_authorized_deviations(
        [
            {
                "metric": "object_pairing_consistency",
                "target_ids": ["chair_01", "desk_01"],
                "relation": "chair_faces_away_from_desk",
                "source": "explicit_prompt_requirement",
                "source_claim_id": "oor_003",
            }
        ],
        allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
    )
    assert normalized[0]["source_claim_id"] == "oor_003"
    assert json.loads(json.dumps(normalized)) == normalized

    with pytest.raises(Exception, match="source_claim_id"):
        validate_authorized_deviations(
            [
                {
                    "metric": "style_consistency",
                    "target_ids": ["a"],
                    "relation": "r",
                    "source": "explicit_prompt_requirement",
                    "source_claim_id": "",
                }
            ],
            allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
        )
