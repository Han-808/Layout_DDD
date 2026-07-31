from __future__ import annotations

import json

import pytest

from benchmark.evaluator import (
    CANONICAL_HIERARCHY,
    EVIDENCE_STRATEGIES,
    GROUPING_CONFIG_PATH,
    GROUPING_IMPLEMENTATION,
    GROUPING_POLICY_ID,
    canonical_hierarchy,
    grouping_policy_provenance,
    validate_evidence_plan,
)
from benchmark.evaluator.evidence_contract import (
    EvidenceContractError,
    validate_evidence_strategy,
    validate_global_policy,
    validate_local_policy,
    validate_router_options,
    validate_trigger_states,
)


# --- Canonical hierarchy: one L0--L4 ownership map ----------------------------


def test_canonical_hierarchy_places_oor_oar_in_l2_fine_grained() -> None:
    hierarchy = canonical_hierarchy()
    assert hierarchy["l2_specification_fidelity"]["metrics"] == [
        "oor",
        "oar",
        "functional_semantic_fidelity",
    ]
    assert hierarchy["l2_specification_fidelity"][
        "functional_semantic_components"
    ] == [
        "room_scene_type",
        "visual_functional_intent",
        "required_functional_areas",
        "local_functionality",
    ]
    # And explicitly not in L1.
    assert "oor" not in hierarchy["l1_physical_plausibility"]["metrics"]
    assert "oar" not in hierarchy["l1_physical_plausibility"]["metrics"]
    # Generic physical metrics remain L1.
    for metric in ("collision", "oob", "support"):
        assert metric in hierarchy["l1_physical_plausibility"]["metrics"]
    # It is a copy, not the shared mutable object.
    hierarchy["l1_physical_plausibility"]["metrics"].append("mutation")
    assert "mutation" not in CANONICAL_HIERARCHY["l1_physical_plausibility"]["metrics"]


# --- Grouping provenance ------------------------------------------------------


def test_grouping_policy_provenance_shape() -> None:
    provenance = grouping_policy_provenance()
    assert provenance == {
        "policy_id": "vlm_visual_evidence_scope_v2",
        "implementation": "src/benchmark/grouping/vlm.py",
        "config": "configs/grouping/vlm_visual_evidence_scope_v2.yaml",
        "role": "evidence_partition_not_metric_verdict",
    }
    assert GROUPING_POLICY_ID == "vlm_visual_evidence_scope_v2"
    assert GROUPING_IMPLEMENTATION == "src/benchmark/grouping/vlm.py"
    assert (
        GROUPING_CONFIG_PATH
        == "configs/grouping/vlm_visual_evidence_scope_v2.yaml"
    )
    # JSON round-trips.
    assert json.loads(json.dumps(provenance)) == provenance


# --- Vocabulary validators ----------------------------------------------------


def test_strategy_and_trigger_validation() -> None:
    for strategy in EVIDENCE_STRATEGIES:
        assert validate_evidence_strategy(strategy) == strategy
    with pytest.raises(EvidenceContractError, match="evidence_strategy"):
        validate_evidence_strategy("teleport")
    assert validate_trigger_states(["suspicious", "insufficient_evidence"]) == [
        "suspicious",
        "insufficient_evidence",
    ]
    with pytest.raises(EvidenceContractError):
        validate_trigger_states(["nonsense"])
    with pytest.raises(EvidenceContractError):
        validate_trigger_states([])


def test_global_and_local_policy_validation() -> None:
    validate_global_policy({"view_family": "wall_occlusion_aware_room_perspective", "image_budget": 2, "top_down": False})
    with pytest.raises(EvidenceContractError, match="image_budget"):
        validate_global_policy({"view_family": "x", "image_budget": 0, "top_down": False})
    with pytest.raises(EvidenceContractError, match="top_down"):
        validate_global_policy({"view_family": "x", "image_budget": 1, "top_down": "no"})

    validate_local_policy({"camera_scope": "group_local", "grouping_policy_id": GROUPING_POLICY_ID})
    validate_local_policy(
        {
            "camera_scope": "group_local",
            "grouping_policy_id": GROUPING_POLICY_ID,
            "activation_condition": "prompt_specified_local_functionality",
            "trigger_states": ["prompt_specified_local_functionality"],
        }
    )
    with pytest.raises(EvidenceContractError, match="camera_scope"):
        validate_local_policy({"camera_scope": "global", "grouping_policy_id": GROUPING_POLICY_ID})
    with pytest.raises(EvidenceContractError, match="grouping_policy_id"):
        validate_local_policy({"camera_scope": "object_local", "grouping_policy_id": ""})
    with pytest.raises(EvidenceContractError, match="activation_condition"):
        validate_local_policy(
            {
                "camera_scope": "group_local",
                "grouping_policy_id": GROUPING_POLICY_ID,
                "activation_condition": "all_groups",
            }
        )


def test_router_options_reject_execution() -> None:
    validate_router_options(
        {
            "global_screen_then_local": {"router": "vlm_global_screen", "trigger_states": ["suspicious"]},
            "script_screen_then_local": {
                "router": "canonical_l3_scale_candidate_router",
                "trigger_states": ["statistical_outlier", "unknown_coverage"],
                "executes_router": False,
            },
        }
    )
    with pytest.raises(EvidenceContractError, match="executes_router"):
        validate_router_options({"script_screen_then_local": {"router": "r", "executes_router": True}})
    with pytest.raises(EvidenceContractError, match="must be one of"):
        validate_router_options({"not_a_strategy": {"router": "r"}})


# --- Evidence plan validation -------------------------------------------------


def test_evidence_plan_requires_prompt_context() -> None:
    # A plan whose text_context omits original_prompt is rejected.
    with pytest.raises(EvidenceContractError, match="original_prompt"):
        validate_evidence_plan(
            {"evidence_strategy": "global_only", "text_context": ["asset_policy"]}
        )
    # A complete plan validates and is returned.
    plan = {
        "evidence_strategy": "global_screen_then_local",
        "global_policy": {"view_family": "wall_occlusion_aware_room_perspective", "image_budget": 2, "top_down": False},
        "local_policy": {"camera_scope": "group_local", "grouping_policy_id": GROUPING_POLICY_ID, "trigger_states": ["suspicious"]},
        "text_context": ["original_prompt", "parsed_required_functional_areas"],
    }
    assert validate_evidence_plan(plan) is plan


def test_missing_routing_evidence_never_valid_semantics() -> None:
    # The vocabulary encodes that only suspicious/insufficient_evidence request
    # local, and failed stays failed; there is no "valid" default token.
    from benchmark.evaluator.evidence_contract import (
        LOCAL_REQUESTING_STATES,
        ROUTER_STATES,
    )

    assert set(LOCAL_REQUESTING_STATES) == {"suspicious", "insufficient_evidence"}
    assert "failed" in ROUTER_STATES
    assert "valid" not in ROUTER_STATES
