from __future__ import annotations

import json
import math

import pytest

from benchmark.api.evaluation import (
    _resolve_run_scoring_profile,
    _strict_scoring_profile_score,
)
from benchmark.evaluator.profile import L1, resolve_evaluation_profile
from benchmark.evaluator.scoring import (
    INTRINSIC_VALIDITY_PROFILE_ID,
    L3_METRIC_WEIGHTS,
    PROMPT_CONDITIONED_QUALITY_PROFILE_ID,
    project_metric_events,
    resolve_scoring_profile,
    score_collision_report,
    score_l3_metric_report,
    score_oob_report,
    score_support_report,
    scoring_reliability_summary,
    scoring_profile_for_run,
)


def _ids(count: int = 10) -> tuple[str, ...]:
    return tuple(f"obj_{index}" for index in range(count))


def _event(
    event_id: str,
    *,
    burden: float,
    allocations: dict[str, float],
) -> dict:
    return {
        "event_id": event_id,
        "burden": burden,
        "allocations": allocations,
    }


def test_versioned_scoring_profiles_are_explicit_and_normalized() -> None:
    intrinsic = resolve_scoring_profile(INTRINSIC_VALIDITY_PROFILE_ID)
    prompted = resolve_scoring_profile(PROMPT_CONDITIONED_QUALITY_PROFILE_ID)

    assert intrinsic["layer_weights"] == {
        "l1_physical_plausibility": 0.30,
        "l2_specification_fidelity": 0.0,
        "l3_scene_quality": 0.70,
        "l4_downstream_task_functionality": 0.0,
    }
    assert prompted["layer_weights"] == {
        "l1_physical_plausibility": 0.20,
        "l2_specification_fidelity": 0.20,
        "l3_scene_quality": 0.60,
        "l4_downstream_task_functionality": 0.0,
    }
    assert math.isclose(sum(L3_METRIC_WEIGHTS.values()), 1.0)
    assert scoring_profile_for_run(has_l2_task=False)[
        "scoring_profile_id"
    ] == INTRINSIC_VALIDITY_PROFILE_ID
    assert scoring_profile_for_run(has_l2_task=True)[
        "scoring_profile_id"
    ] == PROMPT_CONDITIONED_QUALITY_PROFILE_ID


def test_leaderboard_profile_never_renormalizes_around_missing_l2() -> None:
    prompted = resolve_scoring_profile(
        PROMPT_CONDITIONED_QUALITY_PROFILE_ID
    )
    reports = {
        "l1_physical_plausibility": {"status": "evaluated", "score": 0.8},
        "l2_specification_fidelity": {
            "status": "not_applicable",
            "score": None,
        },
        "l3_scene_quality": {"status": "evaluated", "score": 0.9},
        "l4_downstream_task_functionality": {
            "status": "not_implemented",
            "score": None,
        },
    }

    assert _strict_scoring_profile_score(
        reports,
        prompted["layer_weights"],
    ) is None


def test_detector_override_does_not_silently_restore_old_profile_weights() -> None:
    resolved = resolve_evaluation_profile(
        {
            L1: {
                "metric_config": {
                    "collision": {"separation_threshold_m": 0.05}
                }
            }
        }
    )
    selected = _resolve_run_scoring_profile(
        scoring_profile_id=None,
        active_l2_metrics=[],
        resolved_profile=resolved,
    )

    assert selected is not None
    assert selected["scoring_profile_id"] == INTRINSIC_VALIDITY_PROFILE_ID


def test_coefficient_and_worst_event_floor_examples() -> None:
    ids = _ids()
    scale_max = project_metric_events(
        "scale_consistency",
        ordered_object_ids=ids,
        events=[_event("scale", burden=1.0, allocations={ids[0]: 1.0})],
    )
    functional_max = project_metric_events(
        "functional_consistency",
        ordered_object_ids=ids,
        events=[_event("function", burden=1.0, allocations={ids[0]: 1.0})],
    )
    scale_mild = project_metric_events(
        "scale_consistency",
        ordered_object_ids=ids,
        events=[_event("scale-mild", burden=0.4, allocations={ids[0]: 0.4})],
    )
    functional_mild = project_metric_events(
        "functional_consistency",
        ordered_object_ids=ids,
        events=[_event("function-mild", burden=0.4, allocations={ids[0]: 0.4})],
    )

    assert scale_max["score"] == 0.7
    assert functional_max["score"] == 0.75
    assert scale_mild["score"] == 0.88
    assert functional_mild["score"] == 0.9


def test_relation_split_preserves_one_event_burden_and_unsplit_floor() -> None:
    ids = _ids()
    result = project_metric_events(
        "functional_consistency",
        ordered_object_ids=ids,
        events=[
            _event(
                "relation",
                burden=1.0,
                allocations={ids[0]: 0.5, ids[1]: 0.5},
            )
        ],
    )

    assert result["burden_total_b_m"] == 1.0
    assert result["p_max"] == 1.0
    assert result["worst_event_floor_deduction"] == 0.25
    assert result["score"] == 0.75


def test_one_object_is_capped_at_one_burden_per_metric() -> None:
    ids = _ids()
    result = project_metric_events(
        "style_consistency",
        ordered_object_ids=ids,
        events=[
            _event("a", burden=1.0, allocations={ids[0]: 1.0}),
            _event("b", burden=1.0, allocations={ids[0]: 1.0}),
        ],
    )

    assert result["raw_object_burdens"][ids[0]] == 2.0
    assert result["capped_object_burdens"][ids[0]] == 1.0
    assert result["burden_total_b_m"] == 1.0


def test_collision_uses_joint_projected_thickness_and_relation_split() -> None:
    ids = _ids()
    report = {
        "pairs": [
            {
                "object_a": ids[0],
                "object_b": ids[1],
                "final_verdict": "invalid",
                "mesh_evidence": {},
                "scoring_geometry": {
                    "penetration_depth_m": 0.02,
                    "projected_thickness_a_m": 0.10,
                    "projected_thickness_b_m": 0.20,
                },
            }
        ]
    }
    result = score_collision_report(report, ordered_object_ids=ids)

    assert result["events"][0]["magnitude"] == pytest.approx(1.0)
    assert result["events"][0]["allocations"] == {
        ids[0]: 0.5,
        ids[1]: 0.5,
    }


def test_oob_uses_largest_normalized_face_and_floor_tolerance() -> None:
    ids = _ids()
    report = {
        "objects": [
            {
                "object_id": ids[0],
                "final_verdict": "invalid",
                "plane_penetration_m": {
                    "west_oob": 0.0,
                    "east_oob": 0.2,
                    "south_oob": 0.0,
                    "north_oob": 0.0,
                    "floor_oob": 0.01,
                    "ceiling_oob": 0.0,
                },
                "floor_contact_tolerance_m": 0.005,
                "obb_intervals": {
                    "x": [0.0, 1.0],
                    "y": [0.0, 1.0],
                    "z": [-0.01, 0.99],
                },
            }
        ]
    }
    result = score_oob_report(report, ordered_object_ids=ids)

    assert result["events"][0]["magnitude"] == 1.0
    assert result["burden_total_b_m"] == 1.0


def test_support_uses_robust_gap_and_deduplicates_floating_stack() -> None:
    ids = _ids()
    record = lambda object_id: {
        "object_id": object_id,
        "final_verdict": "invalid",
        "positive_clearance_statistics_m": {"p25": 0.10},
        "direct_contact_tolerance_m": 0.01,
        "size_z_m": 1.0,
        "geometry_evidence_degraded": False,
        "grounding_status": "no_reliable_tolerance_contact",
    }
    report = {
        "objects": [record(ids[0]), record(ids[1])],
        "support_contact_graph_edges": [
            {"source_object_id": ids[1], "target_object_id": ids[0]}
        ],
    }
    result = score_support_report(report, ordered_object_ids=ids)

    assert result["event_count"] == 1
    assert result["causal_root_by_invalid_object"] == {
        ids[0]: ids[0],
        ids[1]: ids[0],
    }
    assert result["events"][0]["deduplicated_source_ids"] == [ids[1]]
    assert result["events"][0]["magnitude"] == pytest.approx(0.6)


def test_l3_pairing_relation_is_one_full_burden_split_across_repair_set() -> None:
    ids = _ids()
    report = {
        "judgement": {
            "verdict": "invalid",
            "defects": [
                {
                    "category": "incompatible_object_set",
                    "target_ids": [ids[0], ids[1]],
                    "scope": "group_member_category_compatibility",
                    "relation": "incompatible set",
                    "reason": "No coherent role remains.",
                }
            ],
        }
    }
    result = score_l3_metric_report(
        "object_pairing_consistency", report, ordered_object_ids=ids
    )

    assert result["events"][0]["burden"] == 1.0
    assert result["events"][0]["allocations"] == {
        ids[0]: 0.5,
        ids[1]: 0.5,
    }


def test_missing_l3_severity_uses_minimum_confirmed_invalid_burden() -> None:
    ids = _ids()
    report = {
        "judgement": {
            "verdict": "invalid",
            "defects": [
                {
                    "target_ids": [ids[0]],
                    "scope": "ordinary_static_visual_usability",
                    "relation": "frontage",
                    "reason": "Use is materially impaired.",
                }
            ],
        }
    }
    result = score_l3_metric_report(
        "functional_consistency", report, ordered_object_ids=ids
    )

    assert result["events"][0]["severity"] == "impaired"
    assert result["events"][0]["burden"] == 0.4
    assert result["score"] == 0.9


def test_confidence_never_changes_burden() -> None:
    ids = _ids()
    defect = {
        "category": "style_outlier",
        "severity": "gross",
        "target_ids": [ids[0]],
        "scope": "visible_design_language",
        "relation": "style",
        "reason": "Material outlier.",
    }
    low = score_l3_metric_report(
        "style_consistency",
        {"judgement": {"verdict": "invalid", "confidence": 0.1, "defects": [defect]}},
        ordered_object_ids=ids,
    )
    high = score_l3_metric_report(
        "style_consistency",
        {"judgement": {"verdict": "invalid", "confidence": 0.99, "defects": [defect]}},
        ordered_object_ids=ids,
    )

    assert low == high


def test_repeated_l3_observations_merge_and_keep_stronger_severity() -> None:
    ids = _ids()
    base = {
        "category": "style_outlier",
        "target_ids": [ids[0]],
        "scope": "significant_visible_style_incompatibility",
        "relation": "same visible design-language conflict",
        "reason": "Repeated in global and local evidence.",
    }
    result = score_l3_metric_report(
        "style_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "defects": [
                    {**base, "severity": "noticeable"},
                    {**base, "severity": "gross"},
                ],
            }
        },
        ordered_object_ids=ids,
    )

    assert result["event_count"] == 1
    assert result["events"][0]["severity"] == "gross"
    assert result["events"][0]["observation_count"] == 2
    assert len(result["events"][0]["deduplicated_source_references"]) == 2
    assert result["burden_total_b_m"] == 1.0


def test_stable_check_identity_deduplicates_cross_phase_wording() -> None:
    ids = _ids()
    result = score_l3_metric_report(
        "functional_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "defects": [
                    {
                        "check_id": "functional_check_7",
                        "category": "directed_surface_unusable",
                        "severity": "impaired",
                        "target_ids": [ids[0]],
                        "scope": "scene_global",
                        "relation": "front side seems constrained",
                        "reason": "Observed globally.",
                    },
                    {
                        "check_id": "functional_check_7",
                        "category": "directed_surface_unusable",
                        "severity": "blocked",
                        "target_ids": [ids[0]],
                        "scope": "group_real_world_usability",
                        "relation": "ordinary access is blocked",
                        "reason": "Confirmed locally.",
                    },
                ],
            }
        },
        ordered_object_ids=ids,
    )

    assert result["event_count"] == 1
    assert result["events"][0]["severity"] == "blocked"
    assert result["events"][0]["observation_count"] == 2


def test_stable_check_identity_deduplicates_category_drift() -> None:
    ids = _ids()
    result = score_l3_metric_report(
        "functional_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "defects": [
                    {
                        "check_id": "functional_check_8",
                        "category": "group_function_failure",
                        "severity": "impaired",
                        "target_ids": [ids[0]],
                        "scope": "scene_global",
                        "relation": "ordinary use seems constrained",
                        "reason": "Observed globally.",
                    },
                    {
                        "check_id": "functional_check_8",
                        "category": "directed_surface_unusable",
                        "severity": "blocked",
                        "target_ids": [ids[0]],
                        "scope": "group_real_world_usability",
                        "relation": "the usable side is blocked",
                        "reason": "Confirmed locally.",
                    },
                ],
            }
        },
        ordered_object_ids=ids,
    )

    assert result["event_count"] == 1
    assert result["events"][0]["category"] == "directed_surface_unusable"
    assert result["events"][0]["severity"] == "blocked"
    assert result["events"][0]["observation_count"] == 2


def test_collision_duplicate_and_reversed_pair_is_counted_once() -> None:
    ids = _ids()
    base = {
        "final_verdict": "invalid",
        "mesh_evidence": {"containment_a_in_b": True},
    }
    result = score_collision_report(
        {
            "pairs": [
                {**base, "object_a": ids[0], "object_b": ids[1]},
                {**base, "object_a": ids[1], "object_b": ids[0]},
            ]
        },
        ordered_object_ids=ids,
    )

    assert result["event_count"] == 1
    assert result["events"][0]["observation_count"] == 2
    assert result["events"][0]["affected_object_ids"] == [ids[0], ids[1]]
    assert result["burden_total_b_m"] == 1.0


def test_function_owned_placement_check_is_zero_burden_but_independent_check_scores() -> None:
    ids = _ids()
    result = score_l3_metric_report(
        "semantic_placement_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "placement_check_results": [
                    {
                        "check_id": "placement_function_owned",
                        "conclusion": "excluded_function_owned",
                        "function_event_ref": "function_event_1",
                        "same_physical_event": True,
                    },
                    {
                        "check_id": "placement_independent",
                        "conclusion": "invalid",
                    },
                ],
                "defects": [
                    {
                        "check_id": "placement_independent",
                        "category": "zone_placement_mismatch",
                        "severity": "atypical",
                        "target_ids": [ids[0]],
                        "scope": "scene_zone_placement",
                        "relation": "independent zone mismatch",
                        "reason": "This is distinct from the Function event.",
                    }
                ],
            }
        },
        ordered_object_ids=ids,
    )

    assert result["event_count"] == 1
    assert result["events"][0]["source_reference"]["original_defect"][
        "check_id"
    ] == "placement_independent"


def test_scoring_reliability_publishes_forced_ambiguity_and_failures() -> None:
    summary = scoring_reliability_summary(
        l1_metrics={
            "collision": {
                "status": "checked",
                "budget_exhaustion_forced_choice": {
                    "forced_binary": True,
                    "evidence_ambiguous": True,
                },
            },
            "support": {
                "status": "requires_vlm",
                "adjudication_failures": ["endpoint unavailable"],
            },
        },
        l3_metrics={
            "functional_consistency": {
                "status": "evaluated",
                "affects_score": True,
            }
        },
        judge_episodes=[
            {
                "judge_method": "adjudicate_p0b",
                "metric": "collision",
                "status": "invalid",
                "stop_reason": "budget_exhausted_forced_choice",
                "budget_exhaustion_forced_choice": {
                    "applied": True,
                    "ambiguity_before_forcing": True,
                },
            },
            {
                "judge_method": "adjudicate_p0b",
                "metric": "support",
                "status": "unresolved",
                "budget_exhaustion_forced_choice": {"applied": False},
            },
            {
                "judge_method": "adjudicate_scene_quality",
                "metric": "functional_consistency",
                "status": "valid",
                "budget_exhaustion_forced_choice": {"applied": False},
            },
        ],
    )

    assert summary["active_metric_count"] == 3
    assert summary["judge_episode_count"] == 3
    assert summary["forced_binary_episode_count"] == 1
    assert summary["forced_binary_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["evidence_ambiguous_episode_count"] == 1
    assert summary["evidence_ambiguity_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["unresolved_metric_ids"] == []
    assert summary["terminal_state"] == "infrastructure_failure"
    assert summary["infrastructure_failures"][0]["metric_id"] == (
        "l1_physical_plausibility.support"
    )


def test_scoring_reliability_does_not_deduplicate_identical_judge_episodes() -> None:
    episode = {
        "judge_method": "adjudicate_scene_quality",
        "metric": "style_consistency",
        "status": "valid",
        "budget_exhaustion_forced_choice": {"applied": False},
    }
    summary = scoring_reliability_summary(
        l1_metrics={},
        l3_metrics={
            "style_consistency": {
                "status": "evaluated",
                "affects_score": True,
            }
        },
        judge_episodes=[episode, episode],
    )

    assert summary["judge_episode_count"] == 2
    assert [item["episode_id"] for item in summary["episodes"]] == [
        "judge_episode_0000",
        "judge_episode_0001",
    ]


def test_scoring_reliability_l2_failed_claim_is_infrastructure_failure() -> None:
    summary = scoring_reliability_summary(
        l1_metrics={},
        l2_metrics={
            "oor": {
                "status": "failed",
                "claims": [
                    {
                        "claim_id": "oor_001",
                        "resolution": "failed",
                        "reason": "endpoint failure",
                    }
                ],
            }
        },
        l3_metrics={},
        judge_episodes=[],
        required_metrics_by_layer={
            "l2_specification_fidelity": ["oor"]
        },
    )

    assert summary["terminal_state"] == "infrastructure_failure"
    assert any(
        item.get("claim_id") == "oor_001"
        for item in summary["infrastructure_failures"]
    )


def test_scoring_reliability_l2_unresolved_claim_remains_unresolved() -> None:
    summary = scoring_reliability_summary(
        l1_metrics={},
        l2_metrics={
            "oar": {
                "status": "incomplete",
                "claims": [
                    {
                        "claim_id": "oar_001",
                        "resolution": "unresolved",
                        "reason": "evidence unavailable",
                    }
                ],
            }
        },
        l3_metrics={},
        judge_episodes=[],
        required_metrics_by_layer={
            "l2_specification_fidelity": ["oar"]
        },
    )

    assert summary["terminal_state"] == "unresolved"
    assert summary["unresolved_claims"][0]["claim_id"] == "oar_001"


def test_scoring_reliability_incomplete_score_coverage_cannot_report_complete() -> None:
    summary = scoring_reliability_summary(
        l1_metrics={"collision": {"status": "checked"}},
        l3_metrics={},
        judge_episodes=[],
        required_metrics_by_layer={
            "l1_physical_plausibility": ["collision"]
        },
        scoring_coverage={"complete": False},
    )

    assert summary["terminal_state"] == "unresolved"
    assert "scoring_coverage" in summary["unresolved_metric_ids"]


def test_empty_canonical_denominator_is_not_evaluable() -> None:
    result = project_metric_events(
        "functional_consistency",
        ordered_object_ids=(),
        events=[],
    )

    assert result["n_scene"] == 0
    assert result["score"] is None
    assert result["metric_deduction"] is None


def test_saved_ledger_replays_without_a_judge_call() -> None:
    ids = _ids()
    original = project_metric_events(
        "semantic_placement_consistency",
        ordered_object_ids=ids,
        events=[
            _event(
                "placement",
                burden=0.4,
                allocations={ids[0]: 0.4},
            )
        ],
        nominal_weight=L3_METRIC_WEIGHTS[
            "semantic_placement_consistency"
        ],
    )
    saved = json.loads(json.dumps(original))
    replayed = project_metric_events(
        "semantic_placement_consistency",
        ordered_object_ids=saved["ordered_canonical_object_ids"],
        events=saved["events"],
        nominal_weight=saved["nominal_metric_weight"],
    )

    assert replayed["score"] == original["score"]
    assert replayed["burden_total_b_m"] == original["burden_total_b_m"]
    assert replayed["p_max"] == original["p_max"]


def test_functional_causal_blocker_is_not_automatically_double_charged() -> None:
    ids = _ids()
    result = score_l3_metric_report(
        "functional_consistency",
        {
            "judgement": {
                "verdict": "invalid",
                "defects": [
                    {
                        "category": "approach_clearance_failure",
                        "severity": "blocked",
                        "target_ids": [ids[0]],
                        "causal_object_ids": [ids[1]],
                        "scoring_target_ids": [ids[1]],
                        "scope": "group_real_world_usability",
                        "relation": "external blocker",
                        "reason": "The placed blocker prevents ordinary access.",
                    }
                ],
            }
        },
        ordered_object_ids=ids,
    )

    event = result["events"][0]
    assert event["affected_object_ids"] == [ids[0]]
    assert event["causal_object_ids"] == [ids[1]]
    assert event["scoring_target_ids"] == [ids[1]]
    assert event["allocations"] == {ids[1]: 1.0}
    assert result["burden_total_b_m"] == 1.0
