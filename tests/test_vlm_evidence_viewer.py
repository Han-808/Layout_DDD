from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_vlm_evidence_viewer.py"
SPEC = importlib.util.spec_from_file_location(
    "build_vlm_evidence_viewer",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)


def test_bundled_evidence_url_is_relative_to_the_viewer_document(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"not-a-real-png-but-stable-copy-input")
    bundle_dir = tmp_path / "nested" / "viewer_bundle"
    resolver = VIEWER.EvidenceURLResolver(
        serve_root=tmp_path,
        bundle_dir=bundle_dir,
    )

    url = resolver.url_for(source)

    assert url is not None
    assert url.startswith("evidence/")
    assert not url.startswith("/")
    assert (bundle_dir / url).is_file()


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
    assert (
        "function showScene(index, updateHash = true, shouldScroll = true)"
        in document
    )
    assert "showScene(initialSceneIndex >= 0 ? initialSceneIndex : 0, false, false)" in document


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


def test_viewer_renders_coverage_conditioned_scoring() -> None:
    case_manifest = {
        "final_decision_status": "unresolved",
        "benchmark_score": None,
        "benchmark_score_100": None,
        "benchmark_score_status": "insufficient_metric_coverage",
        "scoring_profile": {
            "scoring_profile_id": "intrinsic_validity_v1",
            "scoring_spec_version": "object_equivalent_burden_v3",
            "layer_weights": {
                "l1_physical_plausibility": 0.3,
                "l3_scene_quality": 0.7,
            },
        },
        "canonical_object_denominator": {
            "n_scene": 2,
            "ordered_object_ids": ["chair", "table"],
        },
        "scoring_reliability": {
            "terminal_state": "unresolved",
            "judge_episode_count": 3,
            "forced_binary_episode_count": 0,
            "evidence_ambiguous_episode_count": 0,
            "unresolved_metric_ids": [
                "l1_physical_plausibility.collision",
                "scoring_coverage",
            ],
            "infrastructure_failures": [],
        },
    }
    l1_report = {
        "status": "incomplete",
        "score": None,
        "partial_score": 0.9,
        "backend_report": {
            "scoring": {
                "metric_weights": {
                    "collision": 1 / 3,
                    "support": 1 / 3,
                    "oob": 1 / 3,
                }
            }
        },
        "metrics": {
            "collision": {
                "status": "requires_vlm",
                "score": None,
                "reason": "Judge endpoint failed.",
            },
            "support": {
                "status": "checked",
                "score": 1.0,
                "scoring": {
                    "event_count": 0,
                    "events": [],
                    "coefficient_n_m": 3,
                    "burden_total_b_m": 0,
                    "p_max": 0,
                    "metric_deduction": 0,
                },
            },
            "oob": {
                "status": "checked",
                "score": 0.8,
                "scoring": {
                    "event_count": 1,
                    "events": [
                        {
                            "category": "room_boundary_crossing",
                            "severity": "major",
                            "magnitude": 0.3,
                            "burden": 0.6,
                            "scoring_target_ids": ["chair"],
                        },
                        {
                            "category": "room_boundary_crossing",
                            "severity": None,
                            "magnitude": 0.3,
                            "burden": 0.6,
                            "scoring_target_ids": ["table"],
                        },
                    ],
                    "coefficient_n_m": 3,
                    "burden_total_b_m": 0.6,
                    "p_max": 0.6,
                    "metric_deduction": 0.2,
                },
            },
        },
    }
    l3_report = {
        "status": "evaluated",
        "score": 1.0,
        "resolved_score": 1.0,
        "coverage": {
            "eligible_count": 5,
            "resolved_count": 5,
            "complete": True,
        },
        "scoring": {
            "metric_weights": {
                "scale_consistency": 0.12,
                "style_consistency": 0.12,
                "object_pairing_consistency": 0.12,
                "functional_consistency": 0.44,
                "semantic_placement_consistency": 0.2,
            }
        },
        "metrics": {
            metric: {
                "status": "evaluated",
                "score": 1.0,
                "judgement": {"verdict": "valid"},
                "scoring": {"event_count": 0, "events": []},
            }
            for metric in (
                "scale_consistency",
                "style_consistency",
                "object_pairing_consistency",
                "functional_consistency",
                "semantic_placement_consistency",
            )
        },
    }
    # Simulate an older run with a numeric Placement score and incomplete score
    # grounding. The current UI keeps the observed score and reduces its
    # effective aggregate weight to the grounded fraction.
    l3_report["metrics"]["semantic_placement_consistency"]["coverage"] = {
        "score_grounding": {"fraction": 0.5, "complete": False}
    }
    summary = VIEWER.case_scoring_summary(
        case_id="N999",
        case_manifest=case_manifest,
        l1_report=l1_report,
        l3_report=l3_report,
        l1_diagnostics={
            "engineering_failures": [
                {
                    "metric": "collision",
                    "route": "vlm_adjudication_failed",
                    "error": "HTTP 429",
                }
            ]
        },
    )

    collision = next(
        item for item in summary["metrics"] if item["metric"] == "collision"
    )
    oob = next(item for item in summary["metrics"] if item["metric"] == "oob")
    function = next(
        item
        for item in summary["metrics"]
        if item["metric"] == "functional_consistency"
    )
    placement = next(
        item
        for item in summary["metrics"]
        if item["metric"] == "semantic_placement_consistency"
    )
    assert summary["combined_score_100"] == pytest.approx(97.5903614458)
    assert summary["combined_status"] == "partial_coverage"
    assert summary["combined_coverage_fraction"] == pytest.approx(0.83)
    assert collision["score"] is None
    assert oob["score"] == 0.8
    assert oob["overall_weight"] == pytest.approx(0.1)
    assert function["overall_weight"] == pytest.approx(0.308)
    assert placement["score"] is None
    assert placement["observed_score"] == 1.0
    assert placement["coverage_fraction"] == 0.5
    assert placement["coverage_threshold_passed"] is False
    l3_layer = next(
        layer for layer in summary["layers"] if layer["layer"] == "L3"
    )
    assert l3_layer["score"] == 1.0
    assert l3_layer["coverage"]["complete"] is False

    rendered = VIEWER.render_case_scoring_dashboard(summary)
    overview = VIEWER.render_run_scoring_overview(
        [summary],
        evaluator_model_label="gpt-5.6-sol",
    )
    assert "Metric scores and combined result" in rendered
    assert "intrinsic_validity_v1" in rendered
    assert "partial_coverage" in rendered
    assert "coverage 83.0%" in rendered
    assert "failed_coverage_threshold" in rendered
    assert "minimum 80%" in rendered
    assert "Missing evidence is excluded" in rendered
    assert "coverage-conditioned layer score" in rendered
    assert "Scoring ledger unavailable because this metric" in rendered
    assert "HTTP 429" in rendered
    assert "room_boundary_crossing" in rendered
    assert "Deduction events and severity" in rendered
    assert "Severity" in rendered
    assert "major" in rendered
    assert "continuous" in rendered
    assert "burden 0.600 / 1.000" in rendered
    assert "Run-wide result" in overview
    assert "30.8%" in overview
    assert "gpt-5.6-sol" in overview
    assert "Official mean / 100" in overview
    assert "1/1 scenes publishable" in overview
    assert "Per-scene score matrix" in overview

    l3_report["metrics"]["semantic_placement_consistency"]["coverage"] = {
        "score_grounding": {"fraction": 0.0, "complete": False}
    }
    below_threshold = VIEWER.case_scoring_summary(
        case_id="N998",
        case_manifest=case_manifest,
        l1_report=l1_report,
        l3_report=l3_report,
    )
    assert below_threshold["combined_coverage_fraction"] == pytest.approx(
        0.76
    )
    assert below_threshold["combined_observed_score_100"] is not None
    assert below_threshold["combined_score_100"] is None
    assert below_threshold["combined_status"] == (
        "failed_coverage_threshold"
    )

    aggregate = VIEWER.run_scoring_aggregate([summary, below_threshold])
    assert aggregate["case_count"] == 2
    assert aggregate["published_case_count"] == 1
    assert aggregate["official_score_100"] is None
    placement_aggregate = next(
        item
        for item in aggregate["metrics"]
        if item["metric"] == "semantic_placement_consistency"
    )
    assert placement_aggregate["published_case_count"] == 0
    assert placement_aggregate["mean_score_100"] is None


def test_viewer_separates_generator_models_from_evaluator_route(
    tmp_path: Path,
) -> None:
    def case_summary(case_id: str, model: str, score: float) -> dict:
        summary = {
            "case_id": case_id,
            "generator_model_label": model,
            "generator_task_id": f"t{case_id[1:]}",
            "combined_score_100": score,
            "combined_observed_score_100": score,
            "combined_coverage_fraction": 0.9,
            "final_decision_status": "resolved",
            "metrics": [],
        }
        for layer, metric, label in VIEWER.SCORING_METRIC_ORDER:
            summary["metrics"].append(
                {
                    "layer": layer,
                    "metric": metric,
                    "label": label,
                    "score": score / 100.0,
                    "observed_score": score / 100.0,
                    "coverage_fraction": 0.9,
                    "overall_weight": 0.1,
                }
            )
        return summary

    summaries = [
        case_summary("S060", "Opus5", 91.0),
        case_summary("S061", "Opus5", 89.0),
        case_summary("S070", "Grok4.6", 80.0),
    ]
    rendered = VIEWER.render_generator_model_performance(summaries)
    aggregates = VIEWER.generator_model_aggregates(summaries)

    assert [item["model_label"] for item in aggregates] == [
        "Opus5",
        "Grok4.6",
    ]
    assert aggregates[0]["official_score_100"] == pytest.approx(90.0)
    assert aggregates[0]["publishable_partial_case_count"] == 2
    assert "Scene scores by generating model" in rendered
    assert "Opus5" in rendered
    assert "Grok4.6" in rendered
    assert "The evaluator model is held fixed" in rendered
    assert "coverage 90.0%" in rendered

    source_root = tmp_path / "dataset" / "S060"
    original_root = tmp_path / "staging" / "t060"
    source_root.mkdir(parents=True)
    original_root.mkdir(parents=True)
    (source_root / "case_manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "task_id": "t060",
                    "namespace": "strict_one_shot",
                    "original_case_root": str(original_root),
                }
            }
        ),
        encoding="utf-8",
    )
    (original_root / "audit_manifest.json").write_text(
        json.dumps({"model_label": "Opus5"}),
        encoding="utf-8",
    )
    metadata = VIEWER.generator_case_metadata(
        {"case_id": "S060", "source_case_root": str(source_root)}
    )
    assert metadata == {
        "model_label": "Opus5",
        "task_id": "t060",
        "source_namespace": "strict_one_shot",
    }


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

    helper = PROJECT_ROOT / "scripts/blender/open_textured_material_preview.py"
    command = f"open -na Blender --args {blend_path} --python {helper}"
    scene_start = document.index('class="scene-page"')
    command_start = document.index('class="blender-launch"', scene_start)
    grouping_start = document.index('class="grouping-output"', scene_start)
    assert command_start < grouping_start
    assert command in document
    assert "Opens in textured Material Preview" in document
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


def test_residual_global_placement_is_visible_after_typed_review() -> None:
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "semantic_placement_consistency": {
                "global_discovery": {
                    "verdict": "valid",
                    "final_metric_verdict": True,
                    "images_used": ["/evidence/global_angle.png"],
                    "request_metadata": {},
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "member_ids": ["chair", "table"],
                        "status": "evaluated",
                        "score": 1.0,
                        "judgement": {
                            "verdict": "valid",
                            "images_used": [
                                "/evidence/global_angle.png",
                                "/evidence/group_local.png",
                            ],
                            "request_metadata": {},
                        },
                    }
                ],
                "residual_global_placement_review": {
                    "verdict": "invalid",
                    "final_metric_verdict": True,
                    "defects": [
                        {
                            "check_type": "scene_zone",
                            "target_ids": ["chair"],
                            "placement_component": (
                                "residual_global_review"
                            ),
                        }
                    ],
                    "request_metadata": {},
                },
                "residual_global_placement_evidence_paths": [
                    "/evidence/global_angle.png",
                    "/evidence/global_top.png",
                    "/evidence/object_ids.png",
                ],
                "residual_global_placement_phase": {
                    "status": "complete"
                },
                "residual_global_placement_context": {
                    "schema_version": "placement_residual_context_v2",
                    "scene_program": {
                        "scene_type": "living_room",
                        "aesthetic_theme_in_scope": False,
                    },
                    "object_inventory": [
                        {
                            "object_id": "chair",
                            "category": "chair",
                            "group_ids": ["group_001"],
                        },
                        {
                            "object_id": "table",
                            "category": "table",
                            "group_ids": ["group_001"],
                        },
                    ],
                    "object_inventory_complete": True,
                },
                "placement_subscore_policy": {
                    "typed_weight": 0.8,
                    "residual_global_review_weight": 0.2,
                },
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert [call["phase"] for call in calls] == [
        "global_discovery",
        "group_local_review",
        "residual_global_placement_review",
    ]
    residual = calls[-1]
    assert residual["workflow_step"] == 3
    assert residual["images"] == [
        "/evidence/global_angle.png",
        "/evidence/global_top.png",
        "/evidence/object_ids.png",
    ]
    assert residual["evidence_packet_audit"]["actual_roles"] == [
        "angled_global",
        "top_down_global",
        "identity_global",
    ]
    assert "residual scene-global synthesis" in residual["prompt"]
    assert residual["routing_details"]["placement_residual_context"][
        "scene_program"
    ]["scene_type"] == "living_room"
    route = VIEWER.render_phase_routes(calls)
    assert "Residual global Placement" in route

    summary = VIEWER.l3_pipeline_summary(report)[0]
    assert summary["residual_global_placement_phase"]["status"] == "complete"
    assert summary["placement_subscore_policy"][
        "residual_global_review_weight"
    ] == 0.2


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


def test_functional_per_check_episodes_are_individually_auditable() -> None:
    seed = [
        "/evidence/global_perspective.png",
        "/evidence/group_local.png",
        "/evidence/usable_side.png",
    ]
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "functional_consistency": {
                "route": "global_discovery_then_group_local",
                "group_local_check_granularity": "per_check",
                "global_discovery": {
                    "verdict": "valid",
                    "final_metric_verdict": True,
                    "images_used": [seed[0]],
                    "request_metadata": {},
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "member_ids": ["chair", "table"],
                        "status": "evaluated",
                        "score": 0.0,
                        "reason": "One atomic check is invalid.",
                        "vlm_invoked": True,
                        "functional_check_granularity": "per_check",
                        "judge_episode_count": 2,
                        "check_episodes": [
                            {
                                "group_id": "group_001",
                                "member_ids": ["chair", "table"],
                                "status": "evaluated",
                                "score": 1.0,
                                "vlm_invoked": True,
                                "functional_check_granularity": "per_check",
                                "functional_check_episode_id": "check_001",
                                "shared_seed_evidence_reused": True,
                                "evidence_paths": seed,
                                "judgement": {
                                    "verdict": "valid",
                                    "confidence": 0.9,
                                    "reason": "The clearance check passes.",
                                    "images_used": seed,
                                    "request_metadata": {},
                                },
                            },
                            {
                                "group_id": "group_001",
                                "member_ids": ["chair", "table"],
                                "status": "evaluated",
                                "score": 0.0,
                                "vlm_invoked": True,
                                "functional_check_granularity": "per_check",
                                "functional_check_episode_id": "check_002",
                                "shared_seed_evidence_reused": True,
                                "evidence_paths": seed,
                                "judgement": {
                                    "verdict": "invalid",
                                    "confidence": 0.8,
                                    "reason": "The relation check fails.",
                                    "images_used": seed,
                                    "request_metadata": {},
                                },
                            },
                        ],
                    }
                ],
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert [call["phase"] for call in calls] == [
        "global_discovery",
        "group_local_review",
        "group_local_review",
    ]
    assert [call["scope"] for call in calls[1:]] == [
        "group_001 · check_001",
        "group_001 · check_002",
    ]
    assert calls[1]["images"] == seed
    assert calls[2]["images"] == seed
    assert calls[1]["routing_details"][
        "shared_seed_evidence_reused"
    ] is True
    assert calls[2]["routing_details"][
        "group_aggregate_result"
    ]["score"] == 0.0
    route = VIEWER.render_phase_routes(calls)
    assert "1 group(s)" in route
    assert "2 Judge episode(s)" in route
    summary = VIEWER.l3_pipeline_summary(report)[0]
    assert summary["group_local_granularity"] == "per_check"
    assert summary["group_judge_episode_count"] == 2


def test_viewer_exposes_shared_group_bank_reuse_and_window_history() -> None:
    fixed = [
        "path:/evidence/global.png",
        "path:/evidence/group-local.png",
    ]
    shared = "path:/evidence/shared-detail.png"
    window = {
        "schema_version": "bounded_evidence_window_v1",
        "policy": "shared_group_bank",
        "group_id": "group_001",
        "check_id": "check_002",
        "max_active_images": 6,
        "fixed_artifact_ids": fixed,
        "initial_artifact_ids": fixed,
        "final_artifact_ids": [*fixed, shared],
        "events": [
            {
                "trigger": "shared_bank_reuse",
                "reused_artifact_ids": [shared],
                "evicted_artifact_ids": [],
                "overflow_flush_applied": False,
                "camera_selector_invoked": False,
            }
        ],
        "physical_artifacts_deleted": False,
    }
    report = {
        "metric_prompt_version": VIEWER.L3_METRIC_PROMPT_VERSION,
        "metrics": {
            "functional_consistency": {
                "route": "global_discovery_then_group_local",
                "group_local_check_granularity": "per_check",
                "group_local_evidence_policy": "shared_group_bank",
                "group_local_active_window_max_images": 6,
                "functional_group_evidence_bank": {
                    "policy": "shared_group_bank",
                    "groups": {
                        "group_001": {
                            "artifacts": [
                                {
                                    "artifact_id": shared,
                                    "sources": [
                                        {
                                            "source_kind": (
                                                "check_camera_render"
                                            ),
                                            "source_check_id": "check_001",
                                        }
                                    ],
                                    "consumer_check_ids": ["check_002"],
                                }
                            ]
                        }
                    },
                },
                "group_results": [
                    {
                        "group_id": "group_001",
                        "member_ids": ["chair", "table"],
                        "status": "evaluated",
                        "score": 1.0,
                        "check_episodes": [
                            {
                                "group_id": "group_001",
                                "member_ids": ["chair", "table"],
                                "status": "evaluated",
                                "score": 1.0,
                                "vlm_invoked": True,
                                "functional_check_granularity": "per_check",
                                "functional_check_episode_id": "check_002",
                                "evidence_paths": [
                                    "/evidence/global.png",
                                    "/evidence/group-local.png",
                                ],
                                "camera_control_audit": {
                                    "audit": {
                                        "evidence_window": window,
                                        "functional_soft_evidence_contract": {
                                            "schema_version": "functional_soft_evidence_contract_v1",
                                            "active": True,
                                            "camera_selector_review_count": 2,
                                            "camera_selector_acquire_count": 1,
                                            "camera_selector_passed": True,
                                            "usable_side_fallback_applied": True,
                                            "terminal_limited_evidence": False,
                                            "decision_authority": "none",
                                        },
                                        "trace": [
                                            {
                                                "stage": "functional_evidence_readiness",
                                                "status": "acquire",
                                                "decision_authority": "none",
                                            },
                                            {
                                                "stage": "functional_evidence_readiness",
                                                "status": "pass",
                                                "decision_authority": "none",
                                            },
                                        ],
                                    }
                                },
                                "judgement": {
                                    "verdict": "valid",
                                    "confidence": 0.9,
                                    "reason": "The check is satisfied.",
                                    "request_metadata": {},
                                },
                            }
                        ],
                    }
                ],
            }
        },
    }

    calls = VIEWER.l3_calls(report)

    assert len(calls) == 1
    call = calls[0]
    assert call["images"] == [
        "/evidence/global.png",
        "/evidence/group-local.png",
        "/evidence/shared-detail.png",
    ]
    details = call["functional_group_evidence_window"]
    assert details["reused_artifact_ids"] == [shared]
    assert details["camera_selector_avoided_by_bank_reuse"] is True
    assert [
        event["status"]
        for event in details["evidence_readiness_events"]
    ] == ["acquire", "pass"]
    assert details["functional_soft_evidence_contract"][
        "camera_selector_passed"
    ] is True
    assert details["artifact_records"][0]["sources"][0][
        "source_check_id"
    ] == "check_001"
    assert details["artifact_records"][0]["consumer_check_ids"] == [
        "check_002"
    ]
    rendered = VIEWER.render_functional_group_evidence_window(details)
    assert "Shared group evidence window" in rendered
    assert "Bank reuse avoided a CameraSelector call" in rendered
    assert "Soft evidence loop" in rendered
    assert "acquire → pass" in rendered
    assert "usable-side fallback: yes" in rendered
    assert "check_001" in rendered
    assert "check_002" in rendered


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
                "functional_measurement_bank": {
                    "schema_version": "functional_measurement_bank_v1",
                    "generation_stage": (
                        "accepted_checks_before_camera_scheduling"
                    ),
                    "measurement_role": "deterministic_context",
                    "decision_authority": "none",
                    "accepted_check_count": 1,
                    "check_measurement_count": 1,
                    "coverage": {
                        "record_coverage_complete": True,
                    },
                    "check_measurements": [
                        {
                            "check_id": "functional_check_001",
                            "check_type": "clearance",
                            "target_ids": ["chair"],
                            "status": "complete",
                            "measurement_extensions": {},
                            "decision_authority": "none",
                        }
                    ],
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
                "functional_spatial_context": {
                    "schema_version": "functional_spatial_context_v3",
                    "context_role": "attention_only",
                    "decision_authority": "none",
                    "clearance_requirements": [
                        {
                            "object_id": "chair",
                            "observation_goal": "inspect ordinary clearance",
                            "source_check_id": "approach_clearance_01",
                            "ownership": "neutral_prerequisite",
                            "measurement": None,
                        }
                    ],
                    "related_pairs": [],
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
    assert functional["discovery"]["functional_measurement_bank"][
        "check_measurement_count"
    ] == 1
    assert placement["discovery"]["status"] == "complete"
    assert placement["discovery"]["planned"] == 2
    assert placement["discovery"]["functional_spatial_context"][
        "clearance_requirements"
    ][0]["object_id"] == "chair"

    rendered = VIEWER.render_l3_pipeline_summary(report)
    assert "functional_probe_planner_failed" in rendered
    assert "2 candidate(s)" in rendered
    assert "Function → Placement attention context" in rendered
    assert "Pre-camera Measurement Bank · 1 checks" in rendered
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
