from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_vlm_evidence_viewer.py"
SPEC = importlib.util.spec_from_file_location(
    "build_vlm_evidence_viewer",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)


def test_viewer_renders_one_scene_page_with_button_navigation(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "cases" / "N002").mkdir(parents=True)
    (run_root / "cases" / "N003").mkdir(parents=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    for case_id in ("N002", "N003"):
        (run_root / "cases" / case_id / "case_run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "l1_status": "evaluated",
                    "l3_status": "evaluated",
                    "final_decision_status": "resolved",
                }
            ),
            encoding="utf-8",
        )

    output = VIEWER.build_viewer(run_root, serve_root=tmp_path)
    document = output.read_text(encoding="utf-8")

    assert document.count('class="scene-page"') == 2
    assert 'data-scene="N002">' in document
    assert 'data-scene="N003" hidden>' in document
    assert 'data-scene-target="N002"' in document
    assert 'data-scene-target="N003"' in document
    assert 'id="previous-scene"' in document
    assert 'id="next-scene"' in document
    assert 'button.setAttribute("aria-current", "page")' in document
    assert "function showScene(index, updateHash = true)" in document


def test_viewer_renders_an_in_progress_run_without_case_directory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_manifest.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )

    output = VIEWER.build_viewer(run_root, serve_root=tmp_path)
    document = output.read_text(encoding="utf-8")

    assert "No scene report is available yet" in document
    assert "<span>running</span>" in document
    assert 'sceneCounter.textContent = "0 / 0"' in document


def test_viewer_renders_optional_audit_graph_summary(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "N021"
    graph_root = case_dir / "audit_graphs"
    graph_root.mkdir(parents=True)
    (graph_root / "relation_candidate_graph.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "cross_group",
                        "state": "candidate",
                        "sources": [
                            {"source_kind": "vlm_hypothesis"}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (graph_root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "decision_authority": "none",
                "relation_candidate_graph": {
                    "path": "relation_candidate_graph.json",
                },
                "evaluation_query_graphs": [
                    {
                        "metric": "functional_consistency",
                        "node_count": 12,
                        "edge_count": 18,
                        "node_kind_counts": {
                            "typed_check": 2,
                            "check_result": 2,
                            "ownership_event": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rendered = VIEWER.render_audit_graphs(case_dir)

    assert "Evaluation audit graphs" in rendered
    assert "audit only · no decision authority" in rendered
    assert "vlm_hypothesis 1" in rendered
    assert "functional_consistency 1" in rendered
    assert "typed checks</span>" in rendered
    assert "<strong>2</strong><span>typed checks" in rendered
    assert "<strong>1</strong><span>ownership events" in rendered


def test_viewer_renders_typed_placement_and_ownership_chain() -> None:
    metric_report = {
        "placement_check_ledger": {
            "schema_version": "placement_check_ledger_v1",
            "checks": [
                {
                    "check_id": "placement_check_pendant_table",
                    "check_type": "contextual_anchor",
                    "subject_id": "pendant",
                    "context_ids": ["table"],
                    "target_ids": ["pendant"],
                    "owner_stage": "group_local",
                    "owning_group_id": "dining",
                    "origin": "placement_discovery",
                    "lifecycle_status": "resolved",
                    "observation_status": "observed",
                    "check_conclusion": "invalid",
                    "judge_result_ref": "group_local_review:dining",
                }
            ],
            "accepted_check_count": 1,
            "decision_authority": "none",
        },
        "placement_check_coverage": {
            "schema_version": "placement_check_results_v1",
            "required_check_count": 1,
            "resolved_check_count": 1,
            "resolved_check_ids": [
                "placement_check_pendant_table"
            ],
            "unresolved_check_ids": [],
            "complete": True,
            "decision_authority": "none",
        },
        "functional_ownership_ledger": {
            "schema_version": "functional_ownership_ledger_v1",
            "events": [
                {
                    "event_id": "functional_event:chair_blocker",
                    "affected_object_ids": ["wardrobe"],
                    "causal_object_ids": ["chair"],
                    "scoring_target_ids": ["chair"],
                }
            ],
        },
        "cross_metric_ownership_audit": {
            "schema_version": "cross_metric_ownership_audit_v1",
            "excluded_placement_checks": [],
        },
    }

    summary = VIEWER.metric_check_chain_summary(
        "semantic_placement_consistency",
        metric_report,
    )
    rendered = VIEWER.render_metric_check_chain(summary)

    assert "Placement typed-check chain" in rendered
    assert "1/1 resolved · complete" in rendered
    assert "contextual_anchor" in rendered
    assert "pendant" in rendered
    assert "table" in rendered
    assert "group_local · dining" in rendered
    assert "Functional causal ownership · 1 event(s)" in rendered
    assert "affected wardrobe" in rendered
    assert "causal chair" in rendered


def test_viewer_shows_blender_command_at_top_of_each_scene_page(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    case_root = tmp_path / "dataset" / "N016"
    blend_path = case_root / "prepared" / "evaluation.blend"
    blend_path.parent.mkdir(parents=True)
    blend_path.write_bytes(b"blend")
    (case_root / "case_manifest.json").write_text(
        json.dumps({"paths": {"blend": "prepared/evaluation.blend"}}),
        encoding="utf-8",
    )
    case_dir = run_root / "cases" / "N016"
    case_dir.mkdir(parents=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (case_dir / "case_run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_case_root": str(case_root),
            }
        ),
        encoding="utf-8",
    )

    output = VIEWER.build_viewer(run_root, serve_root=tmp_path)
    document = output.read_text(encoding="utf-8")

    command = f"open -a Blender {blend_path}"
    scene_start = document.index('class="scene-page"')
    command_start = document.index('class="blender-launch"', scene_start)
    grouping_start = document.index('class="grouping-output"', scene_start)
    assert command_start < grouping_start
    assert command in document
    assert 'class="copy-blender-command"' in document
    assert 'navigator.clipboard.writeText(button.dataset.copyText)' in document
    assert "function scrollActiveSceneToTop()" in document
    assert "pageTop - controlsHeight - 8" in document


def test_acquisition_timeline_tracks_each_repair_round() -> None:
    request = {
        "target_ids": ["chair", "table"],
        "missing_observations": ["contact_visibility"],
        "view_goal": "show the contact region",
    }
    audit = {
        "stop_reason": "judge_conclusion",
        "audit": {
            "rounds_used": 1,
            "selector_calls_used": 1,
            "trace": [
                {
                    "stage": "evidence_gate",
                    "evidence_round": 0,
                    "result": {"ready": True},
                    "images_used": ["/evidence/initial.png"],
                },
                {
                    "stage": "judge",
                    "evidence_round": 0,
                    "result": {
                        "status": "need_more_evidence",
                        "reason": "Contact is occluded.",
                        "evidence_request": request,
                    },
                    "images_used": ["/evidence/initial.png"],
                },
                {
                    "stage": "acquisition_planner",
                    "evidence_round": 0,
                    "evidence_request": request,
                },
                {
                    "stage": "camera_selector",
                    "selection_stage": "deterministic",
                    "evidence_round": 1,
                    "status": "selected",
                    "result": {"selected_view_ids": ["contact_closeup"]},
                },
                {
                    "stage": "render",
                    "selection_stage": "deterministic",
                    "evidence_round": 1,
                    "status": "completed",
                    "rendered_view_count": 1,
                    "packet_changed": True,
                    "images_used": [
                        "/evidence/initial.png",
                        "/evidence/contact_closeup.png",
                    ],
                },
                {
                    "stage": "judge",
                    "evidence_round": 1,
                    "result": {
                        "status": "valid",
                        "reason": "The new view resolves the contact.",
                    },
                    "images_used": [
                        "/evidence/initial.png",
                        "/evidence/contact_closeup.png",
                    ],
                },
            ],
        },
    }

    timeline = VIEWER.acquisition_timeline(
        control_audit=audit,
        fallback_images=["/evidence/initial.png"],
        final_result={"status": "valid"},
    )

    summary = timeline["summary"]
    assert timeline["trace_source"] == "camera_control_audit.audit.trace"
    assert summary["additional_evidence"] is True
    assert summary["judge_calls"] == 2
    assert summary["judge_request_count"] == 1
    assert summary["selector_calls"] == 1
    assert summary["evidence_rounds"] == 1
    assert summary["added_image_count"] == 1
    assert summary["rejudged"] is True
    render_step = next(
        step for step in timeline["steps"] if step["stage"] == "render"
    )
    assert render_step["new_images"] == [
        "/evidence/contact_closeup.png"
    ]


def test_acquisition_timeline_marks_direct_decision() -> None:
    timeline = VIEWER.acquisition_timeline(
        control_audit={
            "audit": {
                "rounds_used": 0,
                "trace": [
                    {
                        "stage": "evidence_gate",
                        "evidence_round": 0,
                        "result": {"ready": True},
                        "images_used": ["/evidence/initial.png"],
                    },
                    {
                        "stage": "judge",
                        "evidence_round": 0,
                        "result": {"status": "invalid"},
                        "images_used": ["/evidence/initial.png"],
                    },
                ],
            }
        },
        fallback_images=["/evidence/initial.png"],
        final_result={"status": "invalid"},
    )

    assert timeline["summary"]["additional_evidence"] is False
    assert timeline["summary"]["judge_calls"] == 1
    assert timeline["summary"]["evidence_rounds"] == 0
    assert timeline["summary"]["added_image_count"] == 0


def test_acquisition_timeline_reconstructs_missing_trace() -> None:
    timeline = VIEWER.acquisition_timeline(
        control_audit={},
        fallback_images=["/evidence/final.png"],
        final_result={"verdict": "valid"},
    )

    assert timeline["trace_source"] == "reconstructed"
    assert [step["stage"] for step in timeline["steps"]] == [
        "evidence_gate",
        "judge",
    ]
    assert all(step["reconstructed"] for step in timeline["steps"])


def test_prefer_runner_usage_includes_selector_calls_without_cards() -> None:
    reconstructed = {
        "by_role": {
            "judge": {"api_calls_number": 38},
            "camera_selector": {"api_calls_number": 0},
        },
        "source": "reconstructed",
    }
    runner = {
        "api_usage": {
            "by_role": {
                "judge": {
                    "api_calls_number": 39,
                    "tokens_usage": {"total_tokens": 162_663},
                },
                "camera_selector": {
                    "api_calls_number": 1,
                    "tokens_usage": {"total_tokens": 26_831},
                },
                "grouping": {"api_calls_number": 2},
            }
        }
    }

    usage = VIEWER.prefer_runner_usage(runner, reconstructed)

    assert usage["source"] == "summary.api_usage.by_role"
    assert usage["by_role"]["judge"]["api_calls_number"] == 39
    assert usage["by_role"]["camera_selector"]["api_calls_number"] == 1
    assert (
        usage["by_role"]["camera_selector"]["tokens_usage"]["total_tokens"]
        == 26_831
    )


def test_l3_calls_show_global_discovery_before_group_local_review() -> None:
    report = {
        "metric_prompt_version": "l3_metric_global_group_routing_v5",
        "metrics": {
            "functional_consistency": {
                "global_discovery": {
                    "verdict": "valid",
                    "confidence": 0.9,
                    "reason": "Global scene is usable.",
                    "final_metric_verdict": True,
                    "images_used": [
                        "/evidence/global_angle.png",
                        "/evidence/global_top.png",
                    ],
                    "request_metadata": {
                        "call_type": (
                            "vlm_judge.canonical.functional_consistency"
                        ),
                    },
                },
                "global_camera_control_audit": {
                    "audit": {"trace": []}
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "member_ids": ["chair", "table"],
                        "status": "evaluated",
                        "score": 1.0,
                        "vlm_invoked": True,
                        "judgement": {
                            "verdict": "valid",
                            "confidence": 0.8,
                            "reason": "Local arrangement is usable.",
                            "images_used": [
                                "/evidence/global_angle.png",
                                "/evidence/group_local.png",
                            ],
                            "request_metadata": {
                                "call_type": (
                                    "vlm_judge.canonical."
                                    "functional_consistency"
                                ),
                            },
                        },
                    }
                ],
            }
        }
    }

    calls = VIEWER.l3_calls(report)

    assert [call["phase"] for call in calls] == [
        "global_discovery",
        "group_local_review",
    ]
    assert [call["workflow_step"] for call in calls] == [1, 2]
    assert calls[0]["scope"] == "scene global"
    assert calls[0]["images"] == [
        "/evidence/global_angle.png",
        "/evidence/global_top.png",
    ]
    assert calls[1]["images"] == [
        "/evidence/global_angle.png",
        "/evidence/group_local.png",
    ]
    assert calls[0]["evidence_packet_audit"]["status"] == "historical"
    assert calls[0]["evidence_packet_audit"]["actual_image_count"] == 2
    assert calls[0]["evidence_packet_audit"][
        "historical_evidence_preserved"
    ] is True
    assert calls[1]["evidence_packet_audit"]["actual_image_count"] == 2
    route = VIEWER.render_phase_routes(calls)
    assert "Global discovery" in route
    assert "Group-local review" in route


def test_current_functional_packets_are_one_global_then_global_plus_local() -> None:
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "functional_consistency": {
                "global_discovery": {
                    "verdict": "valid",
                    "confidence": 0.9,
                    "reason": "Global scene is usable.",
                    "final_metric_verdict": True,
                    "images_used": ["/evidence/global_perspective.png"],
                    "request_metadata": {},
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "member_ids": ["chair", "table"],
                        "status": "evaluated",
                        "score": 1.0,
                        "vlm_invoked": True,
                        "judgement": {
                            "verdict": "valid",
                            "confidence": 0.8,
                            "reason": "Local arrangement is usable.",
                            "images_used": [
                                "/evidence/global_perspective.png",
                                "/evidence/group_local.png",
                            ],
                            "request_metadata": {},
                        },
                    }
                ],
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert calls[0]["evidence_packet_audit"] == {
        "status": "current_default",
        "run_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "current_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "actual_image_count": 1,
        "actual_roles": ["angled_global"],
        "expected_current_default_image_count": 1,
        "expected_current_default_roles": ["angled_global"],
        "matches_current_default": True,
        "historical_evidence_preserved": True,
    }
    assert calls[1]["evidence_packet_audit"]["status"] == (
        "current_default"
    )
    assert calls[1]["evidence_packet_audit"]["actual_roles"] == [
        "angled_global_context",
        "group_local",
    ]


def test_l3_calls_keep_required_group_scope_when_judge_not_invoked() -> None:
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "semantic_placement_consistency": {
                "global_discovery": {
                    "verdict": "valid",
                    "final_metric_verdict": True,
                    "images_used": ["/evidence/global.png"],
                    "request_metadata": {},
                },
                "group_results": [
                    {
                        "group_id": "group_002",
                        "member_ids": ["cabinet", "stove"],
                        "status": "unresolved",
                        "reason": "group_local_render_evidence_unavailable",
                        "vlm_invoked": False,
                        "evidence_paths": ["/evidence/global.png"],
                        "evidence_resolution": {
                            "scope_satisfied": False,
                            "provider_status": "not_configured",
                        },
                    }
                ],
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert len(calls) == 2
    local = calls[1]
    assert local["phase"] == "group_local_review"
    assert local["scope"] == "group_002"
    assert local["judge_invoked"] is False
    assert local["status"] == "unresolved"
    assert local["acquisition"]["trace_source"] == "judge_not_invoked"
    assert local["acquisition"]["summary"]["judge_calls"] == 0
    assert local["routing_details"]["evidence_resolution"] == {
        "scope_satisfied": False,
        "provider_status": "not_configured",
    }

    overview = VIEWER.acquisition_overview(calls)
    assert overview["totals"]["decisions"] == 1


def test_l3_calls_render_isolated_cross_group_relation_stage() -> None:
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "functional_consistency": {
                "route": (
                    "global_then_cross_group_relations_then_group_local"
                ),
                "status": "evaluated",
                "score": 1.0,
                "global_discovery": {
                    "verdict": "valid",
                    "final_metric_verdict": True,
                    "images_used": ["/evidence/global.png"],
                    "request_metadata": {},
                },
                "cross_group_relation_results": [
                    {
                        "relation_id": "sofa_tv_relation",
                        "target_ids": ["sofa", "television"],
                        "group_ids": ["seating", "media"],
                        "observation_kinds": ["mutual_orientation"],
                        "status": "evaluated",
                        "score": 1.0,
                        "vlm_invoked": True,
                        "evidence_paths": [
                            "/evidence/global.png",
                            "/evidence/sofa_tv.png",
                        ],
                        "judgement": {
                            "verdict": "valid",
                            "confidence": 0.9,
                            "reason": "Facing directions are compatible.",
                            "images_used": [
                                "/evidence/global.png",
                                "/evidence/sofa_tv.png",
                            ],
                            "request_metadata": {},
                        },
                    }
                ],
                "group_results": [
                    {
                        "group_id": "seating",
                        "member_ids": ["sofa", "coffee_table"],
                        "status": "evaluated",
                        "score": 1.0,
                        "vlm_invoked": True,
                        "judgement": {
                            "verdict": "valid",
                            "confidence": 0.8,
                            "reason": "The local group is usable.",
                            "images_used": [
                                "/evidence/global.png",
                                "/evidence/seating.png",
                            ],
                            "request_metadata": {},
                        },
                    }
                ],
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert [call["phase"] for call in calls] == [
        "global_discovery",
        "cross_group_relation_review",
        "group_local_review",
    ]
    relation = calls[1]
    assert relation["workflow_step"] == 2
    assert relation["scope"] == "sofa_tv_relation"
    assert relation["members"] == ["sofa", "television"]
    assert relation["evidence_packet_audit"]["actual_roles"] == [
        "angled_global_context",
        "cross_group_relation_local",
    ]
    assert calls[2]["workflow_step"] == 3

    summary = VIEWER.l3_pipeline_summary(report)[0]
    assert summary["scheduled_relations"] == 1
    assert summary["resolved_relations"] == 1
    assert summary["unresolved_relations"] == 0


def test_l3_pipeline_summary_exposes_discovery_and_unresolved_groups() -> None:
    report = {
        "metrics": {
            "functional_consistency": {
                "route": "global_discovery_then_forced_group_local",
                "status": "unresolved",
                "reason": "one_or_more_required_visual_scopes_unresolved",
                "global_discovery": {
                    "verdict": "ambiguous",
                    "global_status": "insufficient",
                },
                "functional_prejudgement_evidence": {
                    "status": "failed",
                    "selected_judge_probe_paths": [],
                    "budget_usage": {
                        "max_probe_units": 2,
                        "scheduled_probe_count": 0,
                    },
                    "runtime_audit": {
                        "status": "failed",
                        "reason": "functional_probe_planner_failed",
                    },
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "status": "evaluated",
                        "vlm_invoked": True,
                    },
                    {
                        "group_id": "group_002",
                        "status": "unresolved",
                        "vlm_invoked": False,
                    },
                ],
                "group_filter": {
                    "skipped_groups": [{"group_id": "group_003"}],
                },
            },
            "semantic_placement_consistency": {
                "route": "global_discovery_then_forced_group_local",
                "status": "evaluated",
                "score": 1.0,
                "global_discovery": {
                    "verdict": "valid",
                    "global_status": "clear",
                },
                "placement_discovery": {
                    "candidates": [
                        {"subject_id": "lamp"},
                        {"subject_id": "table"},
                    ],
                    "reason": "Two observations routed.",
                },
                "group_results": [],
            },
        }
    }

    summaries = VIEWER.l3_pipeline_summary(report)

    functional = summaries[0]
    placement = summaries[1]
    assert functional["resolved_groups"] == 1
    assert functional["unresolved_groups"] == 1
    assert functional["skipped_groups"] == 1
    assert functional["discovery"]["status"] == "failed"
    assert functional["discovery"]["budget"] == 2
    assert placement["discovery"]["status"] == "complete"
    assert placement["discovery"]["planned"] == 2

    rendered = VIEWER.render_l3_pipeline_summary(report)
    assert "functional_probe_planner_failed" in rendered
    assert "2 candidate(s)" in rendered
    assert "Includes unresolved group scopes" in rendered


def test_functional_evidence_audit_exposes_fail_closed_discovery() -> None:
    grouping = {
        "object_groups": [
            {
                "group_id": "group_001",
                "label": "media",
                "anchor_object_id": "tv_stand",
                "object_ids": ["television", "tv_stand"],
                "reason": "direct interaction and support",
            }
        ],
        "cross_group_relations": [],
    }
    report = {
        "metrics": {
            "functional_consistency": {
                "functional_prejudgement_evidence": {
                    "status": "failed",
                    "decision_authority": "none",
                    "functional_discovery": None,
                    "usable_surface_hypotheses": [],
                    "runtime_audit": {
                        "status": "failed",
                        "reason": "functional_probe_planner_failed",
                        "error_type": "ValueError",
                        "error": (
                            "routine boundary review must not invent a "
                            "boundary goal"
                        ),
                    },
                }
            }
        }
    }
    api_calls = [
        {
            "api_call_number": 20,
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.affordance"
            ),
            "status": "complete",
            "image_count": 1,
            "tokens_usage": {"total_tokens": 3261},
        },
        {
            "api_call_number": 21,
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.relations"
            ),
            "status": "complete",
            "image_count": 1,
            "tokens_usage": {"total_tokens": 1682},
        },
    ]

    audit = VIEWER.functional_evidence_audit(
        grouping=grouping,
        report=report,
        api_calls=api_calls,
    )

    assert audit is not None
    assert audit["status"] == "failed"
    assert audit["fail_closed"] is True
    assert audit["completed_discovery_calls"] == 2
    assert audit["surface_records"] == []
    assert audit["relation_records"] == []
    assert audit["rejected_relation_records"] == []
    assert audit["grouping_relation_records"] == []
    rendered = VIEWER.render_functional_evidence_audit(
        grouping=grouping,
        report=report,
        api_calls=api_calls,
    )
    assert "Usable-side and object–group relationship audit" in rendered
    assert "Object ↔ grouping scope" in rendered
    assert "zero usable-side hypotheses" in rendered
    assert "functional_discovery.affordance" in rendered
    assert "routine boundary review must not invent" in rendered


def test_functional_evidence_audit_renders_accepted_surfaces_and_relations() -> None:
    grouping = {
        "object_groups": [
            {
                "group_id": "group_001",
                "label": "media",
                "anchor_object_id": "tv_stand",
                "object_ids": ["television", "tv_stand"],
                "reason": "direct interaction and support",
            },
            {
                "group_id": "group_002",
                "label": "seating",
                "anchor_object_id": "sofa",
                "object_ids": ["sofa"],
                "reason": "separate local evidence scope",
            },
        ],
        "cross_group_relations": [
            {
                "scope": "cross_group",
                "object_ids": ["television", "tv_stand", "sofa"],
                "group_ids": ["group_001", "group_002"],
                "observation_kinds": ["contextual_affinity"],
                "reason": "Grouping-only scene context.",
            }
        ],
    }
    discovery = {
        "object_affordance_ledger": [
            {
                "object_id": "television",
                "directionality": "directed",
                "surface_roles": ["display_side"],
                "need_clearance": True,
            }
        ],
        "directed_surface_targets": [
            {
                "target_id": "television",
                "surface_roles": ["display_side"],
                "observation_goal": "Show the display side.",
            }
        ],
        "within_group_correspondences": [],
        "cross_group_correspondences": [
            {
                "scope": "cross_group",
                "target_ids": ["television", "sofa"],
                "group_ids": ["group_001", "group_002"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "Show the TV–sofa viewing relation.",
            },
            {
                "scope": "cross_group",
                "target_ids": ["television", "tv_stand", "sofa"],
                "group_ids": ["group_001", "group_002"],
                "observation_kinds": ["mutual_orientation"],
                "observation_goal": "Historical clustered relation.",
            },
        ],
        "provenance": {
            "relation_input_contract": {
                "visual_evidence": [
                    {
                        "role": "scene_global",
                        "path": "/evidence/global.png",
                    },
                    {
                        "role": "global_identity_overlay",
                        "path": "/evidence/identity.png",
                    },
                ],
                "structured_context": {
                    "object_list": [
                        {"id": "television", "category": "television"},
                        {"id": "sofa", "category": "sofa"},
                    ],
                    "trusted_group_partition": grouping["object_groups"],
                },
            }
        },
    }
    report = {
        "metrics": {
            "functional_consistency": {
                "functional_prejudgement_evidence": {
                    "status": "complete",
                    "decision_authority": "none",
                    "functional_discovery": discovery,
                    "usable_surface_hypotheses": [
                        {
                            "target_id": "television",
                            "status": "identified",
                            "surfaces": [
                                {
                                    "surface_role": "display_side",
                                    "side_id": "local_neg_y",
                                }
                            ],
                            "reason": "The screen face is visible.",
                        }
                    ],
                    "runtime_audit": {"status": "complete"},
                }
            }
        }
    }

    audit = VIEWER.functional_evidence_audit(
        grouping=grouping,
        report=report,
        api_calls=[],
    )

    assert audit is not None
    assert audit["fail_closed"] is False
    assert audit["surface_records"][0]["decoded_sides"] == [
        "display_side · local_neg_y"
    ]
    assert audit["relation_records"][0]["target_ids"] == [
        "television",
        "sofa",
    ]
    assert audit["relation_records"][0]["predicate"] == (
        "directional_correspondence"
    )
    assert audit["relation_records"][0]["atomicity"] == "atomic_pair"
    assert len(audit["relation_records"]) == 1
    assert len(audit["rejected_relation_records"]) == 1
    assert audit["rejected_relation_records"][0]["atomicity"] == (
        "legacy_non_atomic_3_objects"
    )
    assert audit["grouping_relation_records"][0]["target_ids"] == [
        "television",
        "tv_stand",
        "sofa",
    ]
    assert audit["relation_input_contract"]["status"] == (
        "persisted_exact"
    )
    rendered = VIEWER.render_functional_evidence_audit(
        grouping=grouping,
        report=report,
        api_calls=[],
    )
    assert "display_side · local_neg_y" in rendered
    assert "television ↔ sofa" in rendered
    assert "directional_correspondence" in rendered
    assert "TV–sofa viewing relation" in rendered
    assert "visual evidence + compact JSON" in rendered
    assert "trusted_group_partition" in rendered
    assert "atomic_pair" in rendered
    assert "Grouping contextual relations · not Functional required checks" in rendered
    assert "Grouping-only scene context." in rendered
    assert "Rejected legacy non-atomic relation records · 1" in rendered
    assert "Historical clustered relation." not in rendered


def test_l3_budget_ui_distinguishes_episode_limit_from_metric_audit() -> None:
    report = {
        "metrics": {
            "functional_consistency": {
                "route": "global_discovery_then_forced_group_local",
                "status": "evaluated",
                "combined_evidence_budget": {
                    "accounting": (
                        "per_judge_episode_limit_with_metric_aggregate_audit"
                    ),
                    "budget_enforcement_scope": "judge_episode",
                    "max_images_per_judge_episode": 6,
                    "metric_aggregate_is_budget_authority": False,
                    "camera_acquisition_ledger": {
                        "total_images_acquired": 15,
                    },
                },
                "global_discovery": {
                    "verdict": "valid",
                    "final_metric_verdict": True,
                    "images_used": ["/evidence/global.png"],
                    "request_metadata": {},
                },
                "group_results": [
                    {
                        "group_id": "group_002",
                        "member_ids": ["sofa", "television"],
                        "status": "evaluated",
                        "score": 1.0,
                        "vlm_invoked": True,
                        "evidence_resolution": {
                            "scope_satisfied": True,
                            "acquisition_budget": {
                                "scope": "group_judge_episode",
                                "max_total_images": 6,
                                "initial_judge_evidence_count": 2,
                                "metric_artifact_count_after": 15,
                            },
                        },
                        "camera_acquisition_episode": {
                            "scope": "group_judge_episode",
                            "ledger_before_judge": {
                                "total_images_acquired": 2,
                            },
                            "ledger_after_judge": {
                                "total_images_acquired": 4,
                            },
                        },
                        "judgement": {
                            "verdict": "valid",
                            "images_used": [
                                "/evidence/global.png",
                                "/evidence/local.png",
                            ],
                            "request_metadata": {},
                        },
                    }
                ],
            }
        }
    }

    calls = VIEWER.l3_calls(report)
    local = calls[1]

    assert local["budget_details"]["resolution_budget"][
        "initial_judge_evidence_count"
    ] == 2
    assert local["routing_details"]["camera_acquisition_episode"][
        "ledger_after_judge"
    ]["total_images_acquired"] == 4
    rendered_call_budget = VIEWER.render_judge_episode_budget(local)
    assert "Judge episode evidence budget" in rendered_call_budget
    assert "2 / 6" in rendered_call_budget
    assert "4 / 6" in rendered_call_budget
    assert "Metric artifact count is telemetry only" in rendered_call_budget

    summaries = VIEWER.l3_pipeline_summary(report)
    assert summaries[0]["budget"] == {
        "accounting": (
            "per_judge_episode_limit_with_metric_aggregate_audit"
        ),
        "scope": "judge_episode",
        "max_images_per_judge_episode": 6,
        "metric_artifact_count": 15,
        "metric_aggregate_is_budget_authority": False,
    }
    rendered_summary = VIEWER.render_l3_pipeline_summary(report)
    assert "Per-Judge episode budget" in rendered_summary
    assert "15 metric artifact(s), audit only" in rendered_summary


def test_object_finding_summary_deduplicates_only_within_metric() -> None:
    global_defect = {
        "scope": "reachability",
        "target_ids": ["chair"],
        "relation": "chair_cannot_reach_table",
        "reason": "The chair cannot serve the table.",
    }
    local_defect = {
        "scope": "orientation_for_use",
        "target_ids": ["chair"],
        "relation": "chair_faces_away",
        "reason": "The same chair faces away locally.",
    }
    report = {
        "metrics": {
            "functional_consistency": {
                "global_discovery": {
                    "verdict": "invalid",
                    "defects": [global_defect],
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "status": "evaluated",
                        "score": 0.0,
                        "judgement": {
                            "verdict": "invalid",
                            "defects": [local_defect],
                        },
                    }
                ],
            },
            "semantic_placement_consistency": {
                "global_discovery": {
                    "verdict": "invalid",
                    "defects": [
                        {
                            **global_defect,
                            "scope": "implausible_local_context",
                            "relation": "chair_misplaced",
                        }
                    ],
                },
                "group_results": [],
            },
        }
    }

    summaries = VIEWER.object_level_finding_summary(report)

    assert [summary["penalty_unit_count"] for summary in summaries] == [1, 1]
    functional = summaries[0]["findings"][0]
    placement = summaries[1]["findings"][0]
    assert functional["object_id"] == placement["object_id"] == "chair"
    assert functional["metric"] == "functional_consistency"
    assert placement["metric"] == "semantic_placement_consistency"
    assert functional["observation_count"] == 2
    assert functional["merged_duplicate_observation_count"] == 1
    assert functional["observed_in_global_and_local"] is True
    assert all(
        summary["cross_metric_deduplication"] is False
        for summary in summaries
    )
