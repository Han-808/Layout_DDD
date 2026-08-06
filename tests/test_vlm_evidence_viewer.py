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
