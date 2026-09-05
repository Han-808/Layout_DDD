from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from benchmark.visual_judge.graphs import (
    build_relation_candidate_graph,
    export_case_audit_graphs,
)
from benchmark.visual_judge.graphs.exporter import (
    _project_relation_lifecycle,
)
from benchmark.visual_judge.interfaces import JudgeRequest


def _grouping() -> dict:
    return {
        "object_catalog": [
            {"object_id": "sofa"},
            {"object_id": "television"},
            {"object_id": "lamp"},
        ],
        "object_groups": [
            {
                "group_id": "seating",
                "object_ids": ["sofa", "lamp"],
            },
            {
                "group_id": "media",
                "object_ids": ["television"],
            },
        ],
    }


def _request(*, members: list[str] | None = None) -> JudgeRequest:
    context = {"case_id": "N021"}
    if members is not None:
        context["group_scope"] = {
            "group_id": "seating",
            "member_ids": members,
        }
    return JudgeRequest(
        task="scene_quality",
        metric="functional_consistency",
        claim_or_event={
            "claim_id": "sofa_tv_facing",
            "target_ids": ["sofa", "television"],
        },
        scene_context={
            "scene_id": "N021",
            "objects": [
                {"id": "sofa"},
                {"id": "television"},
                {"id": "lamp"},
            ],
        },
        deterministic_evidence={"read_only": True},
        visual_evidence=("/tmp/global.png",),
        rubric={"rubric_id": "functional_v1"},
        context=context,
    )


def _report(request: JudgeRequest) -> dict:
    evidence_request = {
        "target_ids": ["sofa", "television"],
        "missing_observations": ["mutual_orientation"],
        "view_goal": "Show both ordinary-use sides.",
        "metadata": {},
    }
    need_more = {
        "status": "need_more_evidence",
        "confidence": 0.4,
        "reason": "The mutual orientation is occluded.",
        "defects": [],
        "evidence_request": evidence_request,
        "backend": "test",
        "provenance": {"model": "test-model"},
    }
    result = {
        "status": "invalid",
        "confidence": 0.9,
        "reason": "The seating side faces away from the display.",
        "defects": [
            {
                "object_ids": ["sofa", "television"],
                "type": "functional_orientation",
            }
        ],
        "evidence_request": None,
        "backend": "test",
        "provenance": {"model": "test-model"},
    }
    audit = {
        "schema_version": "vlm_evaluation_audit_v2",
        "evaluation": {
            "case_id": "N021",
            "metric": "functional_consistency",
        },
        "judge_request": request.to_dict(),
        "trace": [
            {
                "stage": "judge",
                "evidence_round": 0,
                "result": need_more,
                "images_used": ["/tmp/global.png"],
            },
            {
                "stage": "acquisition_planner",
                "episode_index": 1,
                "evidence_round": 0,
                "evidence_request": evidence_request,
            },
            {
                "stage": "camera_selector",
                "episode_index": 1,
                "selection_stage": "deterministic",
                "evidence_round": 1,
                "selected_view_ids": ["relation_wide"],
            },
            {
                "stage": "render",
                "episode_index": 1,
                "selection_stage": "deterministic",
                "evidence_round": 1,
                "result": {
                    "visual_evidence": [
                        {"path": "/tmp/relation_wide.png"}
                    ]
                },
            },
            {
                "stage": "judge",
                "episode_index": 1,
                "evidence_round": 1,
                "result": result,
                "images_used": [
                    "/tmp/global.png",
                    "/tmp/relation_wide.png",
                ],
            }
        ],
    }
    return {
        "metrics": {
            "functional_consistency": {
                "functional_discovery": {
                    "schema_version": "functional_discovery_v1",
                    "inspected_object_ids": [
                        "sofa",
                        "television",
                        "lamp",
                    ],
                    "within_group_correspondences": [],
                    "cross_group_correspondences": [
                        {
                            "discovery_id": "sofa_tv",
                            "target_ids": ["sofa", "television"],
                            "observation_kinds": ["mutual_orientation"],
                            "observation_goal": (
                                "Inspect mutual ordinary-use orientation."
                            ),
                        }
                    ],
                    "provenance": {
                        "backend": "test-discovery",
                        "prompt_version": "test-v1",
                    },
                },
                "global_camera_control_audit": {
                    "judge_method": "adjudicate_scene_quality",
                    "audit": audit,
                },
            }
        }
    }


def test_exporter_writes_stable_connected_graph_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit_graphs"
    first = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=_report(_request()),
        output_dir=output,
    )
    first_bytes = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*.json")
    }
    second = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=_report(_request()),
        output_dir=output,
    )
    second_bytes = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*.json")
    }

    assert first == second
    assert first_bytes == second_bytes
    assert first["status"] == "complete"
    assert first["decision_authority"] == "none"
    assert first["relation_candidate_graph"]["candidate_count"] == 1
    assert len(first["evaluation_query_graphs"]) == 1
    relation = json.loads(
        (output / "relation_candidate_graph.json").read_text(
            encoding="utf-8"
        )
    )
    assert relation["candidates"][0]["state"] == "adjudicated"
    assert relation["candidates"][0]["decision_ref"].startswith(
        "judge_decision:"
    )
    assert relation["candidates"][0]["evidence_refs"] == [
        "/tmp/relation_wide.png"
    ]
    assert (
        relation["metadata"]["lifecycle_authority"]
        == "external_audit_only"
    )
    lifecycle = next(
        iter(relation["metadata"]["lifecycle_projection"].values())
    )
    assert len(lifecycle["evidence_request_refs"]) == 1
    query_path = output / first["evaluation_query_graphs"][0]["path"]
    query = json.loads(query_path.read_text(encoding="utf-8"))
    assert query["decision_authority"] == "none"
    assert any(
        edge["kind"] == "examines_relation"
        for edge in query["edges"]
    )


def test_exporter_merges_identical_mirrored_audits(
    tmp_path: Path,
) -> None:
    report = _report(_request())
    mirrored = deepcopy(
        report["metrics"]["functional_consistency"][
            "global_camera_control_audit"
        ]
    )
    report["metrics"]["functional_consistency"]["judgement"] = {
        "global_camera_control_audit": mirrored,
    }

    result = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=report,
        output_dir=tmp_path / "audit_graphs",
    )

    assert result["status"] == "complete"
    assert len(result["evaluation_query_graphs"]) == 1
    record = result["evaluation_query_graphs"][0]
    assert len(record["source_paths"]) == 2
    assert record["source_path"] == record["source_paths"][0]


def test_exporter_distinguishes_per_check_episodes_in_same_claim(
    tmp_path: Path,
) -> None:
    report = _report(_request())
    metric = report["metrics"]["functional_consistency"]
    first_audit = metric["global_camera_control_audit"]["audit"]
    first_audit["judge_request"]["context"][
        "required_functional_checks"
    ] = [
        {
            "check_id": "functional_check_a",
            "check_type": "directional_correspondence",
            "target_ids": ["sofa", "television"],
            "context_ids": [],
            "owner_stage": "scene_global",
            "owning_group_id": None,
        }
    ]
    second_audit = deepcopy(first_audit)
    second_audit["judge_request"]["context"][
        "required_functional_checks"
    ][0]["check_id"] = "functional_check_b"
    metric["per_check_camera_control_audit"] = {
        "audit": second_audit,
    }

    result = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=report,
        output_dir=tmp_path / "audit_graphs",
    )

    assert result["status"] == "complete"
    records = result["evaluation_query_graphs"]
    assert len(records) == 2
    assert len({record["graph_id"] for record in records}) == 2


def test_query_graph_projects_shared_evidence_window_audit(
    tmp_path: Path,
) -> None:
    report = _report(_request())
    audit = report["metrics"]["functional_consistency"][
        "global_camera_control_audit"
    ]["audit"]
    audit["evidence_window"] = {
        "schema_version": "bounded_evidence_window_v1",
        "policy": "shared_group_bank",
        "group_id": "seating",
        "check_id": "functional_check_sofa_tv",
        "max_active_images": 6,
        "fixed_artifact_ids": [
            "path:/tmp/global.png",
            "path:/tmp/group-local.png",
        ],
        "initial_artifact_ids": [
            "path:/tmp/global.png",
            "path:/tmp/group-local.png",
        ],
        "final_artifact_ids": [
            "path:/tmp/global.png",
            "path:/tmp/group-local.png",
            "path:/tmp/relation_wide.png",
        ],
        "events": [
            {
                "trigger": "shared_bank_reuse",
                "reused_artifact_ids": [
                    "path:/tmp/relation_wide.png"
                ],
                "evicted_artifact_ids": [],
                "overflow_flush_applied": False,
            }
        ],
        "physical_artifacts_deleted": False,
    }

    result = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=report,
        output_dir=tmp_path / "audit_graphs",
    )

    record = result["evaluation_query_graphs"][0]
    graph = json.loads(
        (tmp_path / "audit_graphs" / record["path"]).read_text(
            encoding="utf-8"
        )
    )
    window = graph["metadata"]["evidence_window"]
    assert window["policy"] == "shared_group_bank"
    assert window["bank_reuse_event_count"] == 1
    assert window["reused_artifact_ids"] == [
        "path:/tmp/relation_wide.png"
    ]
    assert window["physical_artifacts_deleted"] is False


def test_unresolved_typed_relation_is_not_marked_adjudicated() -> None:
    relation_graph = build_relation_candidate_graph(
        case_id="N021",
        object_ids=("sofa", "television", "lamp"),
        groups=tuple(_grouping()["object_groups"]),
        functional_discovery={
            "schema_version": "functional_discovery_v3",
            "inspected_object_ids": ["sofa", "television", "lamp"],
            "within_group_correspondences": [],
            "cross_group_correspondences": [
                {
                    "discovery_id": "sofa_tv",
                    "target_ids": ["sofa", "television"],
                    "predicate": "directional_correspondence",
                    "observation_goal": "Observe ordinary-use directions.",
                }
            ],
            "provenance": {"backend": "test", "prompt_version": "v1"},
        },
    )
    candidate_id = relation_graph.candidates[0].candidate_id
    projected = _project_relation_lifecycle(
        relation_graph,
        [
            {
                "graph_id": "query:test",
                "nodes": [
                    {
                        "node_id": "candidate",
                        "kind": "relation_candidate",
                        "attributes": {
                            "candidate_ref": candidate_id,
                            "target_ids": ["sofa", "television"],
                        },
                    },
                    {
                        "node_id": "check",
                        "kind": "typed_check",
                        "attributes": {},
                    },
                    {
                        "node_id": "result",
                        "kind": "check_result",
                        "attributes": {"conclusion": "unresolved"},
                    },
                    {
                        "node_id": "evidence",
                        "kind": "evidence_artifact",
                        "attributes": {"evidence_ref": "/tmp/pair.png"},
                    },
                    {
                        "node_id": "decision",
                        "kind": "decision",
                        "attributes": {"decision_ref": "decision:group"},
                    },
                ],
                "edges": [
                    {
                        "kind": "routes_to_check",
                        "source_id": "candidate",
                        "target_id": "check",
                    },
                    {
                        "kind": "resolves_check",
                        "source_id": "result",
                        "target_id": "check",
                    },
                    {
                        "kind": "check_uses_evidence",
                        "source_id": "check",
                        "target_id": "evidence",
                    },
                ],
            }
        ],
    )

    candidate = projected.candidates[0]
    assert candidate.state == "evidence_acquired"
    assert candidate.evidence_refs == ("/tmp/pair.png",)
    assert candidate.decision_ref is None


def test_exporter_includes_typed_workflow_node_counts(
    tmp_path: Path,
) -> None:
    report = _report(_request())
    metric = report["metrics"]["functional_consistency"]
    metric["functional_check_ledger"] = {
        "schema_version": "functional_check_ledger_v1",
        "checks": [
            {
                "check_id": "functional_check_sofa_tv",
                "check_type": "directional_correspondence",
                "target_ids": ["sofa", "television"],
                "context_ids": [],
                "owner_stage": "scene_global",
                "owning_group_id": None,
                "judge_result_ref": "global_discovery",
                "lifecycle_status": "resolved",
                "observation_status": "observed",
                "check_conclusion": "invalid",
                "result_row": {
                    "check_id": "functional_check_sofa_tv",
                    "target_ids": ["sofa", "television"],
                    "observation_status": "observed",
                    "conclusion": "invalid",
                    "reason": "The functional correspondence fails.",
                },
            }
        ],
    }
    persisted_request = metric["global_camera_control_audit"]["audit"][
        "judge_request"
    ]
    persisted_request["context"]["evidence_phase"] = "global_discovery"
    persisted_request["context"]["required_functional_checks"] = [
        deepcopy(metric["functional_check_ledger"]["checks"][0])
    ]
    metric["functional_ownership_ledger"] = {
        "schema_version": "functional_ownership_ledger_v1",
        "events": [
            {
                "event_id": "functional_event:sofa_tv",
                "affected_object_ids": ["sofa", "television"],
                "cause_kind": "self_layout",
                "causal_object_ids": ["sofa", "television"],
                "scoring_target_ids": ["sofa", "television"],
                "check_refs": ["functional_check_sofa_tv"],
                "decision_ref": "judge:functional:sofa_tv",
                "lifecycle_status": "final",
                "decision_authority": "none",
            }
        ],
    }

    result = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=report,
        output_dir=tmp_path / "audit_graphs",
    )

    assert result["status"] == "complete"
    record = result["evaluation_query_graphs"][0]
    assert record["node_kind_counts"]["typed_check"] == 1
    assert record["node_kind_counts"]["check_result"] == 1
    assert record["node_kind_counts"]["ownership_event"] == 1


def test_exporter_fails_closed_on_scope_partition_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "audit_graphs"
    result = export_case_audit_graphs(
        case_id="N021",
        grouping_report=_grouping(),
        scene_quality_report=_report(
            _request(members=["sofa", "television"])
        ),
        output_dir=output,
    )

    assert result["status"] == "failed"
    assert "trusted partition" in result["error"]
    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json"
    ]
