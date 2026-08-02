from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.aggregate_p0b_visual_evidence_policy import (
    _aligned_rows,
    _mixed_camera_policy_candidates,
    _summary_rows,
    _transition,
)
from scripts.run_p0b_camera_ablation import (
    ARM_CONFIGS,
    EVENT_SCHEMA_VERSION,
    RESUME_CONTRACT_SCHEMA_VERSION,
    _camera_ablation_result_ready,
    _evidence_hashes,
    _filter_evidence_items,
    _gt_events,
    _json_sha256,
    _source_events,
)
from scripts.score_p0b_camera_ablation import _summary_row


def _candidate_request(metric: str, object_ids: list[str]) -> dict:
    event = {"object_ids": object_ids}
    if metric == "collision":
        event.update({"object_a": object_ids[0], "object_b": object_ids[1]})
    else:
        event["object_id"] = object_ids[0]
    return {
        "event": event,
        "natural_language_prompt": "frozen prompt",
        "extracted_relationships": [],
        "detector_evidence": {"backend": "frozen"},
    }


def test_camera_ablation_extracts_only_frozen_gt_events() -> None:
    collision_request = _candidate_request("collision", ["a", "b"])
    oob_request = _candidate_request("oob", ["c"])
    support_request = _candidate_request("support", ["d"])
    report = {
        "reports": {
            "generic_validity": {
                "metrics": {
                    "collision": {
                        "pairs": [
                            {"object_a": "a", "object_b": "b", "judge_result": {"request": collision_request}},
                            {"object_a": "x", "object_b": "y", "judge_result": {"request": {}}},
                        ]
                    },
                    "oob": {"objects": [{"object_id": "c", "judge_result": {"request": oob_request}}]},
                    "support": {"objects": [{"object_id": "d", "judge_result": {"request": support_request}}]},
                }
            }
        }
    }
    gt_events = [
        {"metric": "collision", "event_id": "a|b", "object_ids": ["a", "b"], "label": "invalid"},
        {"metric": "oob", "event_id": "c", "object_ids": ["c"], "label": "invalid"},
        {"metric": "support", "event_id": "d", "object_ids": ["d"], "label": "valid"},
    ]

    extracted = _source_events(report, gt_events)

    assert set(extracted) == {("collision", "a|b"), ("oob", "c"), ("support", "d")}
    assert extracted[("collision", "a|b")]["detector_evidence"] == {"backend": "frozen"}


def test_camera_ablation_summary_treats_invalid_as_positive_class() -> None:
    rows = [
        {"resolved": 1, "match": 1, "gt_label": "invalid", "predicted_label": "invalid"},
        {"resolved": 1, "match": 0, "gt_label": "valid", "predicted_label": "invalid"},
        {"resolved": 1, "match": 0, "gt_label": "invalid", "predicted_label": "valid"},
        {"resolved": 1, "match": 1, "gt_label": "valid", "predicted_label": "valid"},
    ]

    summary = _summary_row(
        "bbox_track",
        "overall",
        rows,
        elapsed_seconds=12.0,
        fallback_events=0,
        degraded_events=0,
    )

    assert summary["accuracy_all"] == 0.5
    assert (summary["tp"], summary["fp"], summary["fn"], summary["tn"]) == (1, 1, 1, 1)


def test_controlled_arms_isolate_camera_highlight_and_global_context() -> None:
    assert ARM_CONFIGS["global_raw"] == {
        "camera_mode": "global_only",
        "evidence_style": "raw",
        "include_overview": True,
    }
    assert ARM_CONFIGS["visibility_raw"]["camera_mode"] == "visibility_ranked"
    assert ARM_CONFIGS["visibility_highlight"]["camera_mode"] == "visibility_ranked"
    assert ARM_CONFIGS["visibility_raw"]["include_overview"] is False
    assert ARM_CONFIGS["visibility_highlight"]["include_overview"] is False


def test_visual_policy_arms_change_only_evidence_selection_policy() -> None:
    fixed = ARM_CONFIGS["fixed_global"]
    deterministic = ARM_CONFIGS["deterministic_metric_local"]
    vlm_select = ARM_CONFIGS["vlm_select_from_candidates"]
    active = ARM_CONFIGS["active_metric_local"]

    assert fixed["camera_mode"] == "global_only"
    assert fixed["include_overview"] is True
    assert (
        deterministic["evidence_style"]
        == vlm_select["evidence_style"]
        == active["evidence_style"]
        == "raw_highlight_global"
    )
    assert deterministic["include_overview"] is vlm_select["include_overview"] is active["include_overview"] is False
    assert deterministic["metric_modes"]["support"] == "support_contact_plane"
    assert vlm_select["metric_modes"]["support"] == "query_cov"
    assert active["metric_modes"]["support"] == "query_cov"
    assert deterministic["active_selector"] is False
    assert vlm_select["active_selector"] is True
    assert vlm_select["max_steps"] == 0
    assert active["active_selector"] is True


def test_camera_ablation_can_freeze_a_metric_specific_event_universe() -> None:
    gt = {
        "events": [
            {"metric": "collision", "event_id": "a|b", "label": "invalid"},
            {"metric": "support", "event_id": "c", "label": "valid"},
        ]
    }

    selected = _gt_events(gt, metrics={"support"})

    assert [event["event_id"] for event in selected] == ["c"]


def test_camera_ablation_resume_requires_exact_successful_contract(tmp_path: Path) -> None:
    identity = {
        "name": "judge",
        "provider": "openai_compatible",
        "endpoint": "http://127.0.0.1:8298/v1",
        "model": "model-a",
        "temperature": 0,
    }
    expected = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "resume_contract_schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "arm": "deterministic_metric_local",
        "camera_mode": "auto",
        "resolved_camera_mode": "visibility_ranked",
        "metric_camera_modes": {"collision": "visibility_ranked"},
        "camera_max_steps": 0,
        "pose_selector_enabled": False,
        "pose_selector_model": None,
        "final_judge_model": identity,
        "evidence_style": "raw_highlight_global",
        "canonical_evidence_style": "raw_highlight_global",
        "metric": "collision",
        "event_id": "a|b",
        "gt_label": "invalid",
        "frozen_event_packet_sha256": "event-a",
        "frozen_input_sha256": {
            "scene": "scene-a",
            "blend_file": "blend-a",
            "judge_config": "judge-a",
            "implementation": {"camera_pose.py": "code-a"},
        },
        "observation_config": {
            "candidate_policy": "legacy",
            "max_views": 2,
            "candidate_count": 6,
        },
    }
    result = {
        **expected,
        "resume_contract_sha256": _json_sha256(expected),
        "predicted_label": "invalid",
        "match": True,
        "judgement": {"verdict": "invalid", "confidence": 0.9},
    }
    evidence_path = tmp_path / "evidence.png"
    evidence_path.write_bytes(b"frozen-evidence-a")
    result["frozen_evidence_sha256"] = _evidence_hashes([], [str(evidence_path)])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert _camera_ablation_result_ready(result_path, expected)

    changed_event = {**expected, "frozen_event_packet_sha256": "event-b"}
    assert not _camera_ablation_result_ready(result_path, changed_event)

    changed_model = deepcopy(expected)
    changed_model["final_judge_model"]["model"] = "model-b"
    assert not _camera_ablation_result_ready(result_path, changed_model)

    changed_observation = deepcopy(expected)
    changed_observation["observation_config"]["max_views"] = 1
    assert not _camera_ablation_result_ready(result_path, changed_observation)

    failed = {**result, "error": "EndpointConnectionError"}
    result_path.write_text(json.dumps(failed), encoding="utf-8")
    assert not _camera_ablation_result_ready(result_path, expected)

    corrupt = {**result, "judgement": {"verdict": "valid"}}
    result_path.write_text(json.dumps(corrupt), encoding="utf-8")
    assert not _camera_ablation_result_ready(result_path, expected)

    result_path.write_text(json.dumps(result), encoding="utf-8")
    evidence_path.write_bytes(b"frozen-evidence-b")
    assert not _camera_ablation_result_ready(result_path, expected)


def test_camera_ablation_filters_same_pose_bundle_by_evidence_style() -> None:
    items = [
        {"path": "global.png", "role": "metric_highlighted_global"},
        {"path": "raw_1.png", "role": "metric_local_rgb", "view_id": "v1"},
        {"path": "highlight_1.png", "role": "metric_local_highlight", "view_id": "v1"},
        {"path": "raw_2.png", "role": "collision_rgb", "view_id": "v2"},
        {"path": "highlight_2.png", "role": "collision_pair_overlay", "view_id": "v2"},
    ]

    raw = _filter_evidence_items(items, "raw")
    # raw_highlight now SUPPLEMENTS raw with the same-pose overlay; it never
    # replaces the raw image (the fixed root cause #5).
    raw_highlight = _filter_evidence_items(items, "raw_highlight")
    raw_highlight_global = _filter_evidence_items(items, "raw_highlight_global")

    assert [item["path"] for item in raw] == ["raw_1.png", "raw_2.png"]
    assert [item["path"] for item in raw_highlight] == [
        "raw_1.png",
        "highlight_1.png",
        "raw_2.png",
        "highlight_2.png",
    ]
    assert [item["path"] for item in raw_highlight_global] == [
        "global.png",
        "raw_1.png",
        "highlight_1.png",
        "raw_2.png",
        "highlight_2.png",
    ]


def test_camera_ablation_highlight_aliases_map_to_raw_supplemented_styles() -> None:
    items = [
        {"path": "raw_1.png", "role": "collision_rgb", "view_id": "v1"},
        {"path": "highlight_1.png", "role": "collision_pair_overlay", "view_id": "v1"},
        {"path": "global.png", "role": "metric_highlighted_global"},
    ]
    # Legacy CLI spellings remain accepted but resolve to the canonical styles
    # that keep the raw image.
    assert [item["path"] for item in _filter_evidence_items(items, "highlight")] == [
        "raw_1.png",
        "highlight_1.png",
    ]
    assert [item["path"] for item in _filter_evidence_items(items, "highlight_global")] == [
        "raw_1.png",
        "highlight_1.png",
        "global.png",
    ]
    # The visibility_highlight arm now carries the canonical raw_highlight style.
    assert ARM_CONFIGS["visibility_highlight"]["evidence_style"] == "raw_highlight"
    assert ARM_CONFIGS["visibility_highlight_global"]["evidence_style"] == "raw_highlight_global"


def test_camera_ablation_summary_reports_accuracy_cost_tradeoff() -> None:
    rows = [
        {
            "resolved": 1,
            "match": 1,
            "gt_label": "invalid",
            "predicted_label": "invalid",
            "image_count": 2,
            "camera_evidence_seconds": 4.0,
            "judge_seconds": 3.0,
            "elapsed_seconds": 7.0,
            "estimated_uncached_seconds": 7.0,
        },
        {
            "resolved": 1,
            "match": 1,
            "gt_label": "valid",
            "predicted_label": "valid",
            "image_count": 2,
            "camera_evidence_seconds": 0.0,
            "judge_seconds": 2.0,
            "elapsed_seconds": 2.0,
            "estimated_uncached_seconds": 6.0,
        },
    ]

    summary = _summary_row(
        "visibility_highlight",
        "overall",
        rows,
        elapsed_seconds=9.0,
        fallback_events=0,
        degraded_events=0,
        camera_mode="visibility_ranked",
        evidence_style="highlight",
    )

    assert summary["accuracy_all"] == 1.0
    assert summary["image_count"] == 4.0
    assert summary["camera_evidence_seconds"] == 4.0
    assert summary["judge_seconds"] == 5.0
    assert summary["estimated_uncached_seconds"] == 13.0


def test_visual_policy_summary_counts_unresolved_as_deployment_failure() -> None:
    rows = [
        {
            "metric": "collision",
            "arm": "fixed_global",
            "gt_label": "invalid",
            "predicted_label": "invalid",
            "resolved": 1,
            "match": 1,
            "error": "",
        },
        {
            "metric": "collision",
            "arm": "fixed_global",
            "gt_label": "invalid",
            "predicted_label": "missing",
            "resolved": 0,
            "match": 0,
            "error": "render_failed",
        },
    ]

    summary = next(
        row
        for row in _summary_rows(rows)
        if row["metric"] == "collision" and row["arm"] == "fixed_global"
    )

    assert summary["deployment_accuracy"] == 0.5
    assert summary["resolved_accuracy"] == 1.0
    assert summary["coverage"] == 0.5


def test_visual_policy_aggregation_can_exclude_active_arm() -> None:
    universe = [
        {
            "case_id": "case_001",
            "base_case_id": "source_001",
            "family": "collision",
            "metric": "collision",
            "fixture_dir": "/fixture",
            "event_id": "a|b",
            "gt_label": "invalid",
            "gt_reason_code": "controlled_transform_gt",
            "gt_status": "usable_routed_event_gt",
        }
    ]
    observed = {
        ("case_001", "collision", "a|b", "fixed_global"): {
            "predicted_label": "invalid",
            "resolved": "1",
        },
        ("case_001", "collision", "a|b", "deterministic_metric_local"): {
            "predicted_label": "invalid",
            "resolved": "1",
        },
    }
    arms = ("fixed_global", "deterministic_metric_local")

    rows = _aligned_rows(universe, observed, arms=arms)
    summaries = _summary_rows(rows, arms=arms)

    assert {row["arm"] for row in rows} == set(arms)
    assert not any(row["arm"] == "active_metric_local" for row in summaries)


def test_visual_policy_transition_separates_accuracy_from_coverage() -> None:
    unresolved = {"resolved": 0, "match": 0}
    correct = {"resolved": 1, "match": 1}
    wrong = {"resolved": 1, "match": 0}

    assert _transition(unresolved, correct) == "unresolved_to_resolved"
    assert _transition(correct, wrong) == "correct_to_wrong"
    assert _transition(wrong, correct) == "wrong_to_correct"


def test_camera_policy_mixtures_are_recombined_offline_without_model_calls() -> None:
    camera_arms = (
        "fixed_global",
        "deterministic_metric_local",
        "vlm_select_from_candidates",
    )
    rows = []
    for metric in ("collision", "oob", "support"):
        for arm in camera_arms:
            rows.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "resolved": 1,
                    "match": int(arm == "deterministic_metric_local"),
                    "gt_label": "invalid",
                    "predicted_label": (
                        "invalid" if arm == "deterministic_metric_local" else "valid"
                    ),
                    "image_count": 5 if arm != "fixed_global" else 2,
                    "camera_evidence_seconds": 1.0,
                    "judge_seconds": 2.0,
                }
            )

    candidates = _mixed_camera_policy_candidates(rows, arms=camera_arms)

    assert len(candidates) == 27
    assert candidates[0]["deployment_accuracy"] == 1.0
    assert all(
        candidates[0][f"{metric}_arm"] == "deterministic_metric_local"
        for metric in ("collision", "oob", "support")
    )
