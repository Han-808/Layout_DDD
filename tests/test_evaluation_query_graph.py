from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.visual_judge.graphs import (
    EvaluationQueryGraph,
    QueryGraphEdge,
    QueryGraphNode,
    build_evaluation_query_graph,
    build_relation_candidate_graph,
)
from benchmark.visual_judge.interfaces import JudgeRequest, JudgeResult


def _judge_request() -> JudgeRequest:
    return JudgeRequest(
        task="scene_quality",
        metric="functional_consistency",
        claim_or_event={
            "claim_id": "functional_group_001",
            "target_ids": ["sofa", "television"],
        },
        scene_context={
            "scene_id": "N021",
            "objects": [
                {"id": "sofa", "category": "sofa"},
                {"id": "television", "category": "television"},
                {"id": "floor_lamp", "category": "floor_lamp"},
            ],
        },
        deterministic_evidence={
            "read_only": True,
            "relative_orientation_available": True,
        },
        visual_evidence=(
            {
                "path": "/tmp/N021_global.png",
                "role": "global_context",
                "evidence_round": 0,
            },
            {
                "path": "/tmp/N021_group_001.png",
                "role": "group_local",
                "evidence_round": 0,
            },
        ),
        rubric={"rubric_id": "functional_consistency_v1"},
        context={
            "case_id": "N021",
            "target_object_ids": ["sofa", "television"],
            "group_scope": {
                "group_id": "group_001",
                "member_ids": ["sofa", "television"],
                "focus_center": [1.0, 2.0, 0.8],
                "extent": [3.0, 2.0, 1.6],
                "grouping_policy_id": "vlm_visual_evidence_scope_v2",
                "grouping_backend": "vlm",
            },
        },
    )


def _need_more_result() -> dict:
    return {
        "status": "need_more_evidence",
        "confidence": 0.45,
        "reason": "Mutual orientation is not visible.",
        "defects": [],
        "evidence_request": {
            "target_ids": ["sofa", "television"],
            "missing_observations": [
                "interaction_side_visible",
                "joint_visibility",
            ],
            "view_goal": (
                "Show both ordinary-use sides and their mutual orientation."
            ),
            "metadata": {},
        },
        "backend": "openai_compatible",
        "provenance": {"model": "test-model"},
    }


def _final_result() -> JudgeResult:
    return JudgeResult.from_value(
        {
            "status": "invalid",
            "confidence": 0.91,
            "reason": "The seating side is directed away from the display.",
            "defects": [
                {
                    "object_ids": ["sofa", "television"],
                    "type": "functional_orientation",
                }
            ],
            "evidence_request": None,
            "backend": "openai_compatible",
            "provenance": {
                "model": "test-model",
                "decision_ref": "decision:N021:functional_group_001",
            },
        }
    )


def _audit() -> dict:
    final = _final_result().to_dict()
    return {
        "schema_version": "vlm_evaluation_audit_v2",
        "evaluation": {
            "case_id": "N021",
            "metric": "functional_consistency",
        },
        "trace": [
            {
                "stage": "evidence_gate",
                "evidence_round": 0,
                "result": {"ready": True},
                "images_used": [
                    "/tmp/N021_global.png",
                    "/tmp/N021_group_001.png",
                ],
            },
            {
                "stage": "judge",
                "evidence_round": 0,
                "result": _need_more_result(),
                "images_used": [
                    "/tmp/N021_global.png",
                    "/tmp/N021_group_001.png",
                ],
            },
            {
                "stage": "functional_evidence_readiness",
                "status": "need_more_evidence",
                "source": "camera_selector",
                "check_id": "functional_group_001",
                "evidence_round": 0,
                "result": {"ready": False},
                "images_used": ["/tmp/N021_global.png"],
                "evidence_request": _need_more_result()[
                    "evidence_request"
                ],
            },
            {
                "stage": "acquisition_planner",
                "episode_index": 1,
                "evidence_round": 0,
                "evidence_request": _need_more_result()[
                    "evidence_request"
                ],
                "camera_constraints": {
                    "target_ids": ["sofa", "television"],
                },
            },
            {
                "stage": "camera_selector",
                "episode_index": 1,
                "selection_stage": "deterministic",
                "evidence_round": 1,
                "outcome": "selected",
                "selected_view_ids": ["functional_relation_wide_01"],
                "backend": "deterministic",
            },
            {
                "stage": "render",
                "selection_stage": "deterministic",
                "evidence_round": 1,
                "status": "completed",
                "result": {
                    "visual_evidence": [
                        {
                            "path": "/tmp/N021_relation_wide.png",
                            "role": "functional_relation",
                            "evidence_round": 1,
                        }
                    ]
                },
                "images_used": [
                    "/tmp/N021_global.png",
                    "/tmp/N021_group_001.png",
                    "/tmp/N021_relation_wide.png",
                ],
            },
            {
                "stage": "evidence_gate",
                "evidence_round": 1,
                "result": {"ready": True},
                "images_used": [
                    "/tmp/N021_global.png",
                    "/tmp/N021_relation_wide.png",
                ],
            },
            {
                "stage": "judge",
                "evidence_round": 1,
                "result": final,
                "images_used": [
                    "/tmp/N021_global.png",
                    "/tmp/N021_relation_wide.png",
                ],
            },
        ],
    }


def _relation_graph():
    return build_relation_candidate_graph(
        case_id="N021",
        object_ids=("sofa", "television", "floor_lamp"),
        groups=(
            {
                "group_id": "group_001",
                "object_ids": ["sofa", "television"],
            },
            {
                "group_id": "group_002",
                "object_ids": ["floor_lamp"],
            },
        ),
        functional_discovery={
            "schema_version": "functional_discovery_v3",
            "inspected_object_ids": [
                "sofa",
                "television",
                "floor_lamp",
            ],
            "within_group_correspondences": [
                {
                    "discovery_id": "functional_correspondence_01",
                    "target_ids": ["sofa", "television"],
                    "observation_kinds": ["mutual_orientation"],
                    "observation_goal": (
                        "Observe ordinary-use sides and mutual orientation."
                    ),
                }
            ],
            "cross_group_correspondences": [],
            "provenance": {
                "backend": "functional_discovery",
                "prompt_version": "functional_relation_audit_v2",
            },
        },
    )


def test_query_graph_projects_complete_evaluation_episode() -> None:
    graph = build_evaluation_query_graph(
        judge_request=_judge_request(),
        judge_result=_final_result(),
        audit=_audit(),
        relation_candidates=_relation_graph(),
    )

    node_kinds = {node.kind for node in graph.nodes}
    edge_kinds = {edge.kind for edge in graph.edges}
    assert {
        "evaluation",
        "metric",
        "claim",
        "scope",
        "object",
        "relation_candidate",
        "evidence_artifact",
        "evidence_request",
        "acquisition_episode",
        "camera_selection",
        "judge_call",
        "decision",
    } <= node_kinds
    assert {
        "examines_relation",
        "requests_evidence",
        "starts_episode",
        "contains_selection",
        "produces_evidence",
        "produces_decision",
    } <= edge_kinds
    assert graph.to_dict()["decision_authority"] == "none"
    assert graph.to_dict()["projection_mode"] == "posthoc_read_only"

    relation_node = next(
        node for node in graph.nodes if node.kind == "relation_candidate"
    )
    assert relation_node.attributes["source_kinds"] == ["vlm_hypothesis"]
    episode = next(
        node for node in graph.nodes if node.kind == "acquisition_episode"
    )
    assert "camera_selector" in episode.attributes["stages"]
    assert "render" in episode.attributes["stages"]


def test_query_graph_projects_typed_check_result_and_ownership_chain() -> None:
    check = {
        "check_id": "placement_check_zone_chair",
        "check_type": "scene_zone",
        "subject_id": "chair",
        "context_ids": [],
        "target_ids": ["chair"],
        "owner_stage": "scene_global",
        "owning_group_id": None,
        "required_observations": [
            "target_visible",
            "global_context_preserved",
            "architecture_plane_visible",
        ],
        "origin": "placement_discovery",
    }
    ownership_event = {
        "event_id": "functional_event:chair_blocker",
        "affected_object_ids": ["wardrobe"],
        "cause_kind": "external_object",
        "causal_object_ids": ["chair"],
        "scoring_target_ids": ["chair"],
        "check_refs": ["functional_check_clearance"],
        "decision_ref": "judge:functional:clearance",
        "lifecycle_status": "final",
        "decision_authority": "none",
    }
    request = JudgeRequest(
        task="scene_quality",
        metric="semantic_placement_consistency",
        claim_or_event={
            "claim_id": "placement_global",
            "target_ids": ["scene"],
        },
        scene_context={
            "scene_id": "N100",
            "objects": [
                {"id": "chair", "category": "chair"},
                {"id": "wardrobe", "category": "wardrobe"},
            ],
        },
        deterministic_evidence={"read_only": True},
        visual_evidence=(
            {
                "path": "/tmp/N100_global.png",
                "role": "global_context",
            },
        ),
        rubric={"rubric_id": "placement_v2"},
        context={
            "case_id": "N100",
            "evidence_phase": "global_discovery",
            "required_placement_checks": [check],
            "functional_ownership_ledger": {
                "schema_version": "functional_ownership_ledger_v1",
                "events": [ownership_event],
            },
        },
    )
    audit = {
        "schema_version": "vlm_evaluation_audit_v2",
        "evaluation": {
            "case_id": "N100",
            "metric": "semantic_placement_consistency",
        },
        "trace": [
            {
                "stage": "judge",
                "evidence_round": 0,
                "result": {
                    "status": "valid",
                    "confidence": 0.9,
                    "reason": (
                        "The exact placement event belongs to Function."
                    ),
                    "defects": [],
                    "evidence_request": None,
                    "backend": "test",
                    "provenance": {
                        "decision_ref": "judge:placement"
                    },
                },
                "images_used": ["/tmp/N100_global.png"],
            }
        ],
    }
    resolved_check = {
        **check,
        "lifecycle_status": "resolved",
        "judge_status": "resolved",
        "judge_result_ref": "global_discovery",
        "observation_status": "observed",
        "check_conclusion": "excluded_function_owned",
        "function_event_ref": ownership_event["event_id"],
    }

    graph = build_evaluation_query_graph(
        judge_request=request,
        audit=audit,
        workflow_context={
            "placement_check_ledger": {
                "schema_version": "placement_check_ledger_v1",
                "checks": [resolved_check],
            },
            "functional_ownership_ledger": {
                "schema_version": "functional_ownership_ledger_v1",
                "events": [ownership_event],
            },
        },
    )

    node_kinds = {node.kind for node in graph.nodes}
    edge_kinds = {edge.kind for edge in graph.edges}
    assert {
        "typed_check",
        "check_result",
        "ownership_event",
    } <= node_kinds
    assert {
        "requires_check",
        "check_targets",
        "check_uses_evidence",
        "produces_check_result",
        "resolves_check",
        "excluded_by_ownership",
        "ownership_affects",
        "ownership_caused_by",
        "ownership_scored_to",
        "supports_decision",
    } <= edge_kinds
    assert graph.metadata["typed_check_count"] == 1
    assert graph.metadata["check_result_count"] == 1
    assert graph.metadata["ownership_event_count"] == 1


def test_functional_relation_candidate_routes_to_exact_typed_check() -> None:
    request = _judge_request().to_dict()
    request["context"]["evidence_phase"] = "group_local_review"
    check = {
        "check_id": "functional_check_sofa_tv_direction",
        "check_type": "directional_correspondence",
        "predicate": "directional_correspondence",
        "target_ids": ["sofa", "television"],
        "owner_stage": "group_local",
        "owning_group_id": "group_001",
        "source_discovery_ids": ["functional_correspondence_01"],
        "routing_discovery_ids": ["functional_correspondence_01"],
    }
    request["context"]["required_functional_checks"] = [check]
    resolved_check = {
        **check,
        "judge_result_ref": "group_local_review:group_001",
        "check_conclusion": "valid",
        "observation_status": "observed",
        "result_row": {
            "check_id": check["check_id"],
            "target_ids": check["target_ids"],
            "observation_status": "observed",
            "conclusion": "valid",
            "reason": "The pair has compatible ordinary-use directions.",
        },
    }

    graph = build_evaluation_query_graph(
        judge_request=request,
        audit=_audit(),
        relation_candidates=_relation_graph(),
        workflow_context={
            "functional_check_ledger": {
                "schema_version": "functional_check_ledger_v5",
                "checks": [resolved_check],
            }
        },
    )

    assert any(edge.kind == "routes_to_check" for edge in graph.edges)
    relation_node = next(
        node for node in graph.nodes if node.kind == "relation_candidate"
    )
    check_node = next(
        node for node in graph.nodes if node.kind == "typed_check"
    )
    assert any(
        edge.kind == "routes_to_check"
        and edge.source_id == relation_node.node_id
        and edge.target_id == check_node.node_id
        for edge in graph.edges
    )


def test_query_graph_fails_closed_on_typed_check_unknown_object() -> None:
    value = _judge_request().to_dict()
    value["context"]["required_placement_checks"] = [
        {
            "check_id": "placement_check_unknown",
            "check_type": "scene_zone",
            "subject_id": "unknown",
            "context_ids": [],
            "target_ids": ["unknown"],
            "owner_stage": "scene_global",
        }
    ]

    with pytest.raises(ValueError, match="invalid object identities"):
        build_evaluation_query_graph(
            judge_request=JudgeRequest.from_value(value),
            audit=_audit(),
            relation_candidates=_relation_graph(),
        )


def test_query_graph_projects_deferred_check_without_current_evidence() -> None:
    request = _judge_request().to_dict()
    request["metric"] = "semantic_placement_consistency"
    request["context"]["required_placement_checks"] = []
    request["context"]["deferred_placement_checks"] = [
        {
            "check_id": "placement_check_deferred_sofa_support",
            "check_type": "support_and_height",
            "subject_id": "sofa",
            "context_ids": [],
            "target_ids": ["sofa"],
            "owner_stage": "group_local",
            "owning_group_id": "group_001",
            "group_ids": ["group_001"],
            "handoff_status": "deferred_to_group_local",
        }
    ]
    audit = _audit()
    audit["evaluation"]["metric"] = "semantic_placement_consistency"

    graph = build_evaluation_query_graph(
        judge_request=request,
        audit=audit,
    )

    typed_node_ids = {
        node.node_id for node in graph.nodes if node.kind == "typed_check"
    }
    assert typed_node_ids
    assert any(
        edge.kind == "defers_check" and edge.target_id in typed_node_ids
        for edge in graph.edges
    )
    assert not any(
        edge.kind == "check_uses_evidence"
        and edge.source_id in typed_node_ids
        for edge in graph.edges
    )


def test_query_graph_projects_target_scope_and_target_local_phase() -> None:
    check = {
        "check_id": "placement_check_chair_context",
        "check_type": "contextual_anchor",
        "subject_id": "chair",
        "context_ids": ["desk"],
        "target_ids": ["chair"],
        "owner_stage": "target_local",
        "owning_group_id": None,
    }
    request = JudgeRequest(
        task="scene_quality",
        metric="semantic_placement_consistency",
        claim_or_event={"claim_id": "target_chair", "target_ids": ["chair"]},
        scene_context={
            "scene_id": "N_target",
            "objects": [
                {"id": "chair", "category": "chair"},
                {"id": "desk", "category": "desk"},
            ],
        },
        deterministic_evidence={"read_only": True},
        visual_evidence=(
            {"path": "/tmp/N_target_global.png", "role": "global_context"},
        ),
        rubric={"rubric_id": "placement_v2"},
        context={
            "case_id": "N_target",
            "evidence_phase": "target_local_confirmation",
            "target_object_ids": ["chair"],
            "context_object_ids": ["desk"],
            "target_scope": {
                "scope_version": "target_camera_scope_v1",
                "scope_id": "target_scope_chair",
                "target_id": "chair",
                "context_ids": ["desk"],
                "framing_ids": ["chair", "desk"],
                "focus_center": [1.0, 1.0, 0.5],
                "extent": [1.5, 1.0, 1.0],
                "require_global_anchor": True,
                "group_identity": None,
                "context_objects_are_defect_owners": False,
            },
        },
    )
    resolved = {
        **check,
        "judge_result_ref": "target_local_confirmation:chair",
        "check_conclusion": "valid",
        "observation_status": "inferred_under_budget",
        "result_row": {
            "check_id": check["check_id"],
            "subject_id": "chair",
            "context_ids": ["desk"],
            "observation_status": "inferred_under_budget",
            "conclusion": "valid",
            "reason": "The retained anchor supports a bounded choice.",
        },
    }

    graph = build_evaluation_query_graph(
        judge_request=request,
        audit={
            "schema_version": "vlm_evaluation_audit_v2",
            "evaluation": {
                "case_id": "N_target",
                "metric": "semantic_placement_consistency",
            },
            "trace": [
                {
                    "stage": "judge",
                    "evidence_round": 0,
                    "result": {
                        "status": "valid",
                        "confidence": 0.4,
                        "reason": "Forced target-local result.",
                        "defects": [],
                        "evidence_request": None,
                        "backend": "test",
                        "provenance": {
                            "decision_ref": "target:chair"
                        },
                    },
                    "images_used": ["/tmp/N_target_global.png"],
                }
            ],
        },
        workflow_context={
            "placement_check_ledger": {
                "schema_version": "placement_check_ledger_v1",
                "checks": [resolved],
            }
        },
    )

    scope = next(node for node in graph.nodes if node.kind == "scope")
    assert scope.attributes["scope_type"] == "target_centered_context"
    assert scope.attributes["target_id"] == "chair"
    assert scope.attributes["context_ids"] == ["desk"]
    assert scope.attributes["context_objects_are_defect_owners"] is False
    assert graph.metadata["typed_check_count"] == 1
    assert graph.metadata["check_result_count"] == 1


def test_query_graph_rejects_target_scope_context_attribution() -> None:
    request = _judge_request().to_dict()
    request["context"].pop("group_scope")
    request["context"].update(
        target_object_ids=["floor_lamp"],
        target_scope={
            "scope_id": "target_scope_sofa",
            "target_id": "sofa",
            "context_ids": ["floor_lamp"],
            "framing_ids": ["sofa", "floor_lamp"],
            "require_global_anchor": True,
            "group_identity": None,
            "context_objects_are_defect_owners": False,
        },
    )

    with pytest.raises(ValueError, match="attribution must remain target-only"):
        build_evaluation_query_graph(judge_request=request)


def test_query_graph_rejects_group_scope_outside_trusted_partition() -> None:
    request = _judge_request().to_dict()
    request["context"]["group_scope"]["member_ids"] = [
        "sofa",
        "floor_lamp",
    ]

    with pytest.raises(ValueError, match="trusted partition"):
        build_evaluation_query_graph(
            judge_request=request,
            audit=_audit(),
            relation_candidates=_relation_graph(),
        )


def test_query_graph_is_deterministic_and_round_trips() -> None:
    kwargs = {
        "judge_request": _judge_request(),
        "judge_result": _final_result(),
        "audit": _audit(),
        "relation_candidates": _relation_graph(),
    }
    first = build_evaluation_query_graph(**kwargs)
    second = build_evaluation_query_graph(**kwargs)

    assert first.to_dict() == second.to_dict()
    assert EvaluationQueryGraph.from_value(first.to_dict()) == first


def test_query_graph_projection_does_not_mutate_inputs() -> None:
    request = _judge_request()
    audit = _audit()
    audit_before = deepcopy(audit)
    request_before = request.to_dict()

    build_evaluation_query_graph(
        judge_request=request,
        judge_result=_final_result(),
        audit=audit,
        relation_candidates=_relation_graph(),
    )

    assert audit == audit_before
    assert request.to_dict() == request_before


def test_query_graph_can_describe_unjudged_initial_query() -> None:
    graph = build_evaluation_query_graph(
        judge_request=_judge_request(),
    )

    assert not any(node.kind == "judge_call" for node in graph.nodes)
    assert not any(node.kind == "decision" for node in graph.nodes)


def test_query_graph_can_recover_final_decision_from_audit_only() -> None:
    graph = build_evaluation_query_graph(
        judge_request=_judge_request(),
        audit=_audit(),
    )

    decision = next(node for node in graph.nodes if node.kind == "decision")
    assert decision.attributes["status"] == "invalid"
    assert decision.attributes["confidence"] == 0.91


def test_query_graph_rejects_relation_graph_for_another_case() -> None:
    relation_graph = build_relation_candidate_graph(
        case_id="N999",
        object_ids=("sofa", "television"),
        groups=(
            {
                "group_id": "group_001",
                "object_ids": ["sofa", "television"],
            },
        ),
    )

    with pytest.raises(ValueError, match="case_id"):
        build_evaluation_query_graph(
            judge_request=_judge_request(),
            relation_candidates=relation_graph,
        )


def test_query_graph_rejects_audit_and_final_result_disagreement() -> None:
    mismatched = _final_result().to_dict()
    mismatched["reason"] = "A different final reason."

    with pytest.raises(ValueError, match="does not match"):
        build_evaluation_query_graph(
            judge_request=_judge_request(),
            judge_result=mismatched,
            audit=_audit(),
        )


def test_query_graph_validates_edge_endpoint_types() -> None:
    evaluation = QueryGraphNode(
        node_id="evaluation:test",
        kind="evaluation",
        label="test",
    )
    metric = QueryGraphNode(
        node_id="metric:test",
        kind="metric",
        label="metric",
    )
    claim = QueryGraphNode(
        node_id="claim:test",
        kind="claim",
        label="claim",
    )
    bad_edge = QueryGraphEdge(
        edge_id="edge:bad",
        kind="contains_metric",
        source_id=evaluation.node_id,
        target_id=claim.node_id,
    )

    with pytest.raises(ValueError, match="cannot connect"):
        EvaluationQueryGraph(
            graph_id="graph:test",
            case_id="N021",
            metric="functional_consistency",
            nodes=(evaluation, metric, claim),
            edges=(bad_edge,),
        )
