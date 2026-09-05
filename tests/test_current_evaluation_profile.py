from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from benchmark.api.evaluation import _strict_scoring_profile_coverage
from benchmark.evaluator.profile import (
    CANONICAL_LAYERS,
    CANONICAL_PROFILE_VERSION,
    L0,
    L1,
    L1_METRICS,
    L2,
    L2_METRICS,
    L3,
    L3_METRICS,
    L4,
    build_evaluation_plan,
    canonical_score_coverage,
    resolve_evaluation_profile,
)
from benchmark.evaluator.specification_fidelity import SpecificationContractError
from benchmark.scoring_profiles import (
    DEFAULT_L3_METRIC_WEIGHTS,
    INTRINSIC_VALIDITY_PROFILE_ID,
    PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID,
    PREVIOUS_L3_METRIC_WEIGHTS,
)
from benchmark.utils.io import load_yaml, read_json
from evaluate import run_evaluate


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PROVIDED_ASSETS = {"mode": "benchmark_provided"}


def _object(
    object_id: str,
    *,
    category: str,
    description: str,
    size: list[float],
    center: list[float],
) -> dict:
    return {
        "id": object_id,
        "jid": f"{object_id}_asset",
        "category": category,
        "description": description,
        "desc": description,
        "size": size,
        "center": center,
        "rotation": [0, 0, 0],
        "asset_ref": {"source_db": "test", "asset_key": f"{object_id}_asset"},
        "asset_proxy": {
            "type": "obb",
            "bbox_center_local": [0, 0, 0],
            "bbox_size": size,
        },
        "metadata": {"interactive": False},
    }


def _scene(*, include_lamp: bool = False) -> dict:
    objects = [
        _object(
            "bed",
            category="bed",
            description="blue velvet bed",
            size=[2.0, 1.6, 0.6],
            center=[2.5, 2.5, 0.3],
        )
    ]
    if include_lamp:
        objects.append(
            _object(
                "lamp",
                category="floor_lamp",
                description="small floor lamp",
                size=[0.4, 0.4, 0.5],
                center=[4.2, 2.5, 0.25],
            )
        )
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "profile_scene",
        "request_id": "profile_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 2.9,
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


def _request(prompt_granularity: str) -> dict:
    return {
        "request_id": "profile_request",
        "instruction": "Place the bed in the room center, left of the lamp.",
        "prompt_granularity": prompt_granularity,
    }


def _relation_contract() -> dict:
    return {
        "contract_version": "specification_contract_v1",
        "source": "benchmark_owned",
        "frozen": True,
        "request_id": "profile_request",
        "claims": {
            "oor": [
                {
                    "claim_id": "oor::bed_left_lamp",
                    "claim_family": "oor",
                    "relation_id": "oor::bed_left_lamp",
                    "relation_type": "left_of",
                    "subject_id": "bed",
                    "object_id": "lamp",
                    "target_ids": ["bed", "lamp"],
                }
            ],
            "oar": [
                {
                    "claim_id": "oar::bed_room_center",
                    "claim_family": "oar",
                    "relation_id": "oar::bed_room_center",
                    "relation_type": "room_center",
                    "subject_id": "bed",
                    "architectural_element": "center_region",
                    "target_ids": ["bed"],
                }
            ],
            "functional_semantic_fidelity": [],
        },
    }


def _run_relation_case(tmp_path: Path, prompt_granularity: str) -> dict:
    return run_evaluate(
        scene=_scene(include_lamp=True),
        out=tmp_path / f"{prompt_granularity}.json",
        scene_request=_request(prompt_granularity),
        specification_contract=_relation_contract(),
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )


def test_default_profile_is_the_exact_frozen_l0_l4_inventory() -> None:
    profile = resolve_evaluation_profile()

    assert profile["profile_version"] == CANONICAL_PROFILE_VERSION
    assert profile["status"] == "frozen"
    assert tuple(layer for layer in profile if layer in CANONICAL_LAYERS) == CANONICAL_LAYERS
    assert profile["layer_weights"] == {
        L1: 0.35,
        L2: 0.25,
        L3: 0.40,
        L4: 0.0,
    }
    assert set(profile[L1]["metrics"]) == set(L1_METRICS)
    assert set(profile[L2]["metrics"]) == set(L2_METRICS)
    assert set(profile[L3]["metrics"]) == set(L3_METRICS)
    assert {
        name: float(config["weight"])
        for name, config in profile[L3]["metrics"].items()
    } == DEFAULT_L3_METRIC_WEIGHTS
    assert profile[L4] == {"enabled": False, "implemented": False, "metrics": {}}

    serialized = repr(profile)
    for removed_category in (
        "prompt_fidelity",
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    ):
        assert removed_category not in profile
        assert f"'{removed_category}':" not in serialized


def test_checked_in_canonical_profile_uses_default_l3_weights() -> None:
    checked_in = load_yaml(
        ROOT / "configs" / "evaluation" / "metric_profile_canonical_v2.yaml"
    )
    assert {
        name: float(config["weight"])
        for name, config in checked_in[L3]["metrics"].items()
    } == DEFAULT_L3_METRIC_WEIGHTS


def test_l1_navigability_and_accessibility_are_frozen_disabled() -> None:
    profile = resolve_evaluation_profile()
    metrics = profile[L1]["metrics"]

    assert metrics["collision"] == {"enabled": True, "weight": pytest.approx(1.0 / 3.0)}
    assert metrics["oob"] == {"enabled": True, "weight": pytest.approx(1.0 / 3.0)}
    assert metrics["support"] == {"enabled": True, "weight": pytest.approx(1.0 / 3.0)}
    assert metrics["navigability"] == {"enabled": False, "weight": 0.0}
    assert metrics["accessibility"] == {"enabled": False, "weight": 0.0}
    assert profile[L1]["never_vlm_metrics"] == ["navigability", "accessibility"]


def test_fine_and_coarse_plans_share_one_contract_and_metric_inventory() -> None:
    active = ["oor", "oar"]
    fine = build_evaluation_plan(
        prompt_granularity="fine_grained",
        active_l2_metrics=active,
    )
    coarse = build_evaluation_plan(
        prompt_granularity="coarse_grained",
        active_l2_metrics=active,
    )

    assert fine["workflow"] == coarse["workflow"] == "canonical_l0_l4"
    assert fine["prompt_granularity_role"] == coarse["prompt_granularity_role"] == "metadata_only"
    assert fine["layer_weights"] == coarse["layer_weights"]
    assert fine["hierarchy"] == coarse["hierarchy"]
    assert fine["layers"] == coarse["layers"]
    assert set(fine["layers"]) == set(CANONICAL_LAYERS)
    assert fine["prompt_granularity"] == "fine_grained"
    assert coarse["prompt_granularity"] == "coarse_grained"
    assert fine["layers"][L2]["metrics"]["oor"]["applicable"] is True
    assert fine["layers"][L2]["metrics"]["oar"]["applicable"] is True
    assert fine["layers"][L2]["metrics"]["functional_semantic_fidelity"]["applicable"] is False


def test_public_generator_structure_cannot_change_prompt_granularity(tmp_path: Path) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "public_structure_granularity.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a bedroom.",
            "prompt_granularity": "coarse_grained",
        },
        object_plan={
            "request_id": "profile_request",
            "scene_type": "bedroom",
            "scene_description": "public generator input",
            "prompt_granularity": "fine_grained",
            "objects": [],
            "global_constraints": [],
            "relations": [],
        },
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )

    assert report["prompt_granularity"] == "coarse_grained"
    assert report["prompt_granularity_role"] == "metadata_only"
    assert "deprecated_inputs_ignored" not in report["evaluation_plan"]


def test_canonical_evaluator_rejects_retired_spatial_ontology(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retired non-game workflow"):
        run_evaluate(
            scene=_scene(),
            out=tmp_path / "retired_spatial_ontology.json",
            scene_request=_request("coarse_grained"),
            spatial_fidelity_ontology={"categories": {}},
            asset_policy=BENCHMARK_PROVIDED_ASSETS,
        )


def test_missing_granularity_defaults_diagnostically_and_invalid_value_is_rejected(
    tmp_path: Path,
) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "default_granularity.json",
        scene_request={
            "request_id": "profile_request",
            "instruction": "Create a bedroom.",
        },
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )
    assert report["prompt_granularity"] == "fine_grained"
    assert report["evaluation_config"]["prompt_granularity_resolution_source"] == "diagnostic_default"

    with pytest.raises(ValueError, match="scene_request.prompt_granularity"):
        run_evaluate(
            scene=_scene(),
            out=tmp_path / "invalid_granularity.json",
            scene_request={
                "request_id": "profile_request",
                "instruction": "Create a bedroom.",
                "prompt_granularity": "auto",
            },
        )


def test_same_contract_executes_oor_and_oar_independently_of_granularity(
    tmp_path: Path,
) -> None:
    fine = _run_relation_case(tmp_path, "fine_grained")
    coarse = _run_relation_case(tmp_path, "coarse_grained")

    for report in (fine, coarse):
        assert report["evaluation_config"]["specification_activation"] == {
            "source": "benchmark_owned_specification_contract",
            "contract_present": True,
            "active_metrics": ["oor", "oar"],
            "prompt_granularity_controls_activation": False,
        }
        assert report["reports"]["oor"]["score"] == 1.0
        assert report["reports"]["oar"]["score"] == 1.0
        assert report["reports"]["specification_fidelity"]["score"] == 1.0
        assert report["layer_reports"][L2]["status"] == "evaluated"
        assert report["layer_reports"][L2]["score"] == 1.0

    assert fine["reports"]["oor"] == coarse["reports"]["oor"]
    assert fine["reports"]["oar"] == coarse["reports"]["oar"]
    assert fine["reports"]["specification_fidelity"]["active_claim_families"] == [
        "oor",
        "oar",
    ]
    assert coarse["reports"]["specification_fidelity"]["active_claim_families"] == [
        "oor",
        "oar",
    ]


def test_vlm_control_override_is_additive_and_records_effective_defaults(
    tmp_path: Path,
) -> None:
    baseline = _run_relation_case(tmp_path, "fine_grained")
    configured = run_evaluate(
        scene=_scene(include_lamp=True),
        out=tmp_path / "vlm_control_override.json",
        scene_request=_request("fine_grained"),
        specification_contract=_relation_contract(),
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
        vlm_evaluation_control={
            "budgets": {
                "max_evidence_rounds": 1,
                "max_total_images": 4,
            }
        },
    )

    control = configured["evaluation_config"]["vlm_evaluation_control"]
    assert control["requested"]["budgets"]["max_evidence_rounds"] == 1
    assert control["effective"]["budgets"]["max_total_images"] == 4
    assert (
        control["effective"]["camera_selector"]["backend"]
        == "deterministic"
    )
    assert (
        control["sources"]["camera_selector.backend"]
        == "fallback_no_existing_selector"
    )
    assert configured["benchmark_score"] == baseline["benchmark_score"]
    assert configured["layer_reports"] == baseline["layer_reports"]
    assert configured["reports"] == baseline["reports"]


def test_canonical_runtime_rejects_retired_presence_count_and_attribute_claims(
    tmp_path: Path,
) -> None:
    contract = {
        "contract_version": "specification_contract_v1",
        "source": "benchmark_owned",
        "frozen": True,
        "claims": {
            "object_presence": [
                {
                    "claim_id": "presence::bed",
                    "claim_family": "object_presence",
                    "target_ids": ["bed"],
                }
            ],
            "object_count": [
                {
                    "claim_id": "count::bed",
                    "claim_family": "object_count",
                    "target_ids": ["bed"],
                }
            ],
            "explicit_attributes": [
                {
                    "claim_id": "attribute::bed",
                    "claim_family": "explicit_attributes",
                    "target_ids": ["bed"],
                }
            ],
        },
    }
    with pytest.raises(SpecificationContractError, match="unknown families"):
        run_evaluate(
            scene=_scene(),
            out=tmp_path / "discarded_legacy_claims.json",
            scene_request=_request("fine_grained"),
            specification_contract=contract,
            asset_policy=BENCHMARK_PROVIDED_ASSETS,
        )


def test_missing_optional_l2_and_ungrounded_l3_fail_score_coverage_threshold(
    tmp_path: Path,
) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "missing_evidence.json",
        scene_request=_request("coarse_grained"),
    )

    assert report["layer_reports"][L1]["score"] == 1.0
    assert report["layer_reports"][L2]["status"] == "incomplete"
    assert report["layer_reports"][L2]["score"] is None
    l3 = report["layer_reports"][L3]
    assert l3["status"] == "incomplete"
    assert l3["score"] is None
    assert report["benchmark_score"] is None
    assert report["benchmark_score_status"] == (
        "failed_coverage_threshold"
    )
    assert report["coverage"]["covered_layers"] == [L1]
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["score_resolution_complete"] is False
    assert report["coverage"]["score_grounding_complete"] is False
    l1_weight = report["scoring_profile"]["layer_weights"][L1]
    assert report["coverage"]["grounded_score_weight"] == pytest.approx(
        l1_weight
    )
    assert report["coverage"]["grounded_score_fraction"] == pytest.approx(
        l1_weight / report["coverage"]["required_weight"]
    )
    assert report["coverage"]["layer_grounding_fractions"] == {
        L1: pytest.approx(1.0),
        L3: pytest.approx(0.0),
    }
    assert all(
        metric["terminal_state"] == "evaluated_degraded"
        and metric["coverage"]["score_grounding"]["fraction"] == 0.0
        for metric in l3["metrics"].values()
    )


def test_profile_coverage_separates_numeric_resolution_from_grounding() -> None:
    l3_grounding = (0.44 * 0.5 + 0.20) / (0.44 + 0.20)
    coverage = canonical_score_coverage(
        {
            L1: {"status": "evaluated", "score": 1.0},
            L3: {
                "status": "evaluated",
                "score": 0.8,
                "active_metrics": [
                    "functional_consistency",
                    "semantic_placement_consistency",
                ],
                "resolved_metrics": [
                    "functional_consistency",
                    "semantic_placement_consistency",
                ],
                "metrics": {
                    "functional_consistency": {
                        "score": 0.7,
                        "weight": 0.44,
                        "coverage": {
                            "score_grounding": {"fraction": 0.5}
                        },
                    },
                    "semantic_placement_consistency": {
                        "score": 1.0,
                        "weight": 0.20,
                        "coverage": {
                            "score_grounding": {"fraction": 1.0}
                        },
                    },
                },
            },
        },
        {
            L1: 0.20,
            L2: 0.0,
            L3: 0.80,
            L4: 0.0,
        },
        profile_version=CANONICAL_PROFILE_VERSION,
    )

    assert coverage["complete"] is True
    assert coverage["score_resolution_complete"] is True
    assert coverage["score_grounding_complete"] is False
    assert coverage["layer_grounding_fractions"][L1] == 1.0
    assert coverage["layer_grounding_fractions"][L3] == pytest.approx(
        l3_grounding
    )
    assert coverage["grounded_score_weight"] == pytest.approx(
        0.20 + 0.80 * l3_grounding
    )


def test_deduction_multiplier_is_part_of_comparability_signature() -> None:
    reports = {
        L1: {"status": "evaluated", "score": 1.0},
        L3: {"status": "evaluated", "score": 1.0},
    }
    weights = {L1: 0.3, L2: 0.0, L3: 0.7, L4: 0.0}
    multiplier_one = canonical_score_coverage(
        reports,
        weights,
        profile_version=CANONICAL_PROFILE_VERSION,
        scoring_profile_id="intrinsic_validity_v1",
        scoring_spec_version="object_equivalent_burden_v3",
        deduction_multiplier=1.0,
    )
    multiplier_two = canonical_score_coverage(
        reports,
        weights,
        profile_version=CANONICAL_PROFILE_VERSION,
        scoring_profile_id="intrinsic_validity_v1",
        scoring_spec_version="object_equivalent_burden_v3",
        deduction_multiplier=2.0,
    )

    assert multiplier_one["comparability_signature"] != multiplier_two[
        "comparability_signature"
    ]
    assert "|deduction_multiplier:1" in multiplier_one[
        "comparability_signature"
    ]
    assert "|deduction_multiplier:2" in multiplier_two[
        "comparability_signature"
    ]


def test_strict_profile_adds_missing_required_layer_to_grounding_denominator() -> None:
    layer_weights = {
        L1: 0.15,
        L2: 0.15,
        L3: 0.70,
        L4: 0.0,
    }
    scoring_reports = {
        L1: {"status": "evaluated", "score": 1.0},
        L2: {"status": "not_applicable", "score": None},
        L3: {"status": "evaluated", "score": 0.8},
        L4: {"status": "not_applicable", "score": None},
    }
    coverage = canonical_score_coverage(
        scoring_reports,
        layer_weights,
        profile_version=CANONICAL_PROFILE_VERSION,
    )
    assert coverage["score_resolution_complete"] is True
    assert coverage["grounded_score_fraction"] == 1.0

    strict = _strict_scoring_profile_coverage(
        coverage,
        scoring_reports=scoring_reports,
        layer_weights=layer_weights,
    )

    assert strict["complete"] is False
    assert strict["score_resolution_complete"] is False
    assert strict["score_grounding_complete"] is False
    assert strict["layer_grounding_fractions"] == {
        L1: 1.0,
        L2: 0.0,
        L3: 1.0,
    }
    assert strict["grounded_score_weight"] == pytest.approx(0.85)
    assert strict["grounded_score_fraction"] == pytest.approx(0.85)


def test_layer_level_grounding_is_authoritative_without_metric_rows() -> None:
    coverage = canonical_score_coverage(
        {
            L1: {
                "status": "evaluated",
                "score": 1.0,
                "coverage": {"score_grounding": {"fraction": 0.25}},
            }
        },
        {L1: 1.0, L2: 0.0, L3: 0.0, L4: 0.0},
        profile_version=CANONICAL_PROFILE_VERSION,
    )

    assert coverage["score_resolution_complete"] is True
    assert coverage["score_grounding_complete"] is False
    assert coverage["grounded_score_weight"] == pytest.approx(0.25)
    assert coverage["grounded_score_fraction"] == pytest.approx(0.25)


def test_canonical_report_has_exact_l0_l4_layers_and_validates_schema(
    tmp_path: Path,
) -> None:
    report = _run_relation_case(tmp_path, "fine_grained")

    assert report["report_schema_version"] == "scene_evaluation_report_v2"
    assert report["profile_version"] == CANONICAL_PROFILE_VERSION
    assert report["workflow"] == "canonical_l0_l4"
    assert tuple(report["layer_reports"]) == CANONICAL_LAYERS
    assert report["category_reports"] == report["layer_reports"]
    assert report["layer_reports"][L0]["status"] == "passed"
    assert report["layer_reports"][L0]["affects_score"] is False
    assert report["layer_reports"][L4] == {
        "layer": L4,
        "status": "not_implemented",
        "score": None,
        "affects_score": False,
        "reason": "downstream_task_type_not_frozen",
        "metrics": {},
    }
    for removed_category in (
        "prompt_fidelity",
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    ):
        assert removed_category not in report["layer_reports"]
        assert removed_category not in report["category_reports"]

    compatibility_keys = {
        "prompt_fidelity",
        "spatial_fidelity",
        "spatial_fidelity_ontology",
        "deprecated_inputs_ignored",
    }
    pending = [report]
    seen_keys: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            seen_keys.update(str(key) for key in value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert compatibility_keys.isdisjoint(seen_keys)

    assert report["coverage"]["active_metrics_by_layer"] == {
        L1: ["collision", "oob", "support"],
        L2: ["oor", "oar"],
        L3: [
            "scale_consistency",
            "object_pairing_consistency",
            "style_consistency",
            "functional_consistency",
            "semantic_placement_consistency",
        ],
        L4: [],
    }
    assert report["coverage"]["resolved_metrics_by_layer"] == {
        L1: ["collision", "oob", "support"],
        L2: ["oor", "oar"],
        L3: [],
        L4: [],
    }
    assert report["coverage"]["active_metric_signatures"] == {
        L1: "collision+oob+support",
        L2: "oor+oar",
        L3: (
            "scale_consistency+object_pairing_consistency+style_consistency+"
            "functional_consistency+semantic_placement_consistency"
        ),
        L4: "none",
    }
    assert report["coverage"]["comparability_signature"].startswith(
        f"{CANONICAL_PROFILE_VERSION}|"
    )
    assert report["coverage"]["case_comparability"] == (
        "compare_only_with_same_profile_version_layer_weight_signature_"
        "and_per_layer_active_metric_signatures"
    )

    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)


def test_report_schema_keeps_v1_l3_active_inventory_to_three_metrics(
    tmp_path: Path,
) -> None:
    report = _run_relation_case(tmp_path, "fine_grained")
    report["profile_version"] = "canonical_scene_evaluation_v1"
    report["evaluation_plan"]["profile_version"] = (
        "canonical_scene_evaluation_v1"
    )
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    validator = Draft202012Validator(schema)
    # A current five-metric report cannot be relabelled as the frozen v1
    # three-metric contract by changing only its version string.
    assert list(validator.iter_errors(report))


def test_runtime_applicability_may_narrow_but_not_enable_frozen_l1_metrics(
    tmp_path: Path,
) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "l1_runtime_applicability.json",
        metric_applicability={
            "support": False,
            "navigability": True,
            "accessibility": True,
        },
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )
    metrics = report["reports"]["generic_validity"]["metrics"]

    assert metrics["support"]["status"] == "not_applicable"
    assert metrics["navigability"]["status"] == "not_applicable"
    assert metrics["accessibility"]["status"] == "not_applicable"
    assert report["evaluation_config"]["metric_applicability"][L1] == {
        "collision": True,
        "oob": True,
        "support": False,
        "navigability": False,
        "accessibility": False,
    }

    with pytest.raises(ValueError, match="unknown metrics"):
        run_evaluate(
            scene=_scene(),
            out=tmp_path / "unknown_metric.json",
            metric_applicability={"teleportation": True},
        )


def test_legacy_support_switch_cannot_disable_canonical_support(
    tmp_path: Path,
) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "legacy_support_switch.json",
        support_enabled=False,
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )

    assert report["evaluation_config"]["metric_applicability"][L1]["support"] is True
    assert report["evaluation_config"]["deprecated_runtime_inputs"][
        "support_enabled"
    ].startswith("ignored")


def test_l1_metric_config_defaults_to_empty_and_accepts_detector_thresholds() -> None:
    profile = resolve_evaluation_profile()
    assert profile[L1]["metric_config"] == {}

    configured = resolve_evaluation_profile(
        {L1: {"metric_config": {"collision": {"separation_threshold_m": 0.05}}}}
    )
    assert configured[L1]["metric_config"] == {
        "collision": {"separation_threshold_m": 0.05}
    }


def test_l1_metric_config_rejects_unknown_and_protocol_owned_keys() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        resolve_evaluation_profile(
            {L1: {"metric_config": {"teleportation": {"eps": 1.0}}}}
        )

    for reserved_key in ("enabled", "official_mode", "detector_only"):
        with pytest.raises(ValueError, match="protocol-owned keys"):
            resolve_evaluation_profile(
                {L1: {"metric_config": {"collision": {reserved_key: True}}}}
            )


def test_l1_oob_threshold_override_reaches_the_detector(tmp_path: Path) -> None:
    sunk = _scene()
    sunk["objects"][0]["center"] = [2.5, 2.5, 0.2]

    default_report = run_evaluate(
        scene=sunk,
        out=tmp_path / "default_oob_threshold.json",
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )
    widened_report = run_evaluate(
        scene=sunk,
        out=tmp_path / "widened_oob_threshold.json",
        evaluation_profile={
            L1: {
                "metric_config": {
                    "oob": {"floor_contact_tolerance_m": 0.25},
                }
            }
        },
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )

    default_oob = default_report["reports"]["generic_validity"]["metrics"]["oob"]
    widened_oob = widened_report["reports"]["generic_validity"]["metrics"]["oob"]
    assert default_oob["candidate_oob_count"] == 1
    assert widened_oob["candidate_oob_count"] == 0
    assert widened_report["evaluation_plan"]["layers"][L1]["metric_config"] == {
        "oob": {"floor_contact_tolerance_m": 0.25}
    }
    assert widened_report["scoring_profile"]["scoring_profile_id"] == (
        INTRINSIC_VALIDITY_PROFILE_ID
    )
    assert widened_report["scoring_profile"]["l3_metric_weights"] == (
        DEFAULT_L3_METRIC_WEIGHTS
    )
    assert widened_report["reports"]["scene_quality"]["scoring"][
        "canonical_profile_metric_weights"
    ] == DEFAULT_L3_METRIC_WEIGHTS
    assert f"scoring_profile:{INTRINSIC_VALIDITY_PROFILE_ID}" in (
        widened_report["coverage"]["comparability_signature"]
    )
    assert "|l3_metric_weights:" in widened_report["coverage"][
        "comparability_signature"
    ]


def test_previous_scoring_profile_replays_frozen_l3_weights(
    tmp_path: Path,
) -> None:
    report = run_evaluate(
        scene=_scene(),
        out=tmp_path / "previous_scoring_profile.json",
        scoring_profile_id=PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID,
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )

    assert report["scoring_profile"]["scoring_profile_id"] == (
        PREVIOUS_INTRINSIC_VALIDITY_PROFILE_ID
    )
    assert report["scoring_profile"]["l3_metric_weights"] == (
        PREVIOUS_L3_METRIC_WEIGHTS
    )
    assert report["reports"]["scene_quality"]["scoring"][
        "canonical_profile_metric_weights"
    ] == PREVIOUS_L3_METRIC_WEIGHTS
    assert "|l3_metric_weights:" in report["coverage"][
        "comparability_signature"
    ]


def test_input_scene_is_not_mutated_by_profile_execution(tmp_path: Path) -> None:
    scene = _scene(include_lamp=True)
    frozen = deepcopy(scene)
    run_evaluate(
        scene=scene,
        out=tmp_path / "scene_immutability.json",
        scene_request=_request("fine_grained"),
        specification_contract=_relation_contract(),
        asset_policy=BENCHMARK_PROVIDED_ASSETS,
    )
    assert scene == frozen
