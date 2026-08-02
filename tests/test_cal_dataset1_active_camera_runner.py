from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cal_dataset1_active_camera",
    ROOT / "scripts" / "run_cal_dataset1_active_camera.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cases() -> list[dict]:
    return MODULE.discover_experiment_cases(
        ROOT / "Support" / "datasets" / "cal_dataset1",
        materialized_roots=[
            ROOT / "Support" / "artifacts" / "outputs" / "exp1_1",
            ROOT / "Support" / "artifacts" / "outputs" / "exp1_1_fine_edge",
        ],
        splits=set(MODULE.SPLITS),
        metrics=set(MODULE.METRICS),
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id="test_active_camera",
        arm=list(MODULE.ARMS),
        candidate_count=5,
        max_steps=2,
        render_engine="BLENDER_WORKBENCH",
        render_width=512,
        render_height=512,
        preview_render_engine="BLENDER_WORKBENCH",
        preview_width=256,
        preview_height=256,
    )


def test_default_plan_discovers_exact_frozen_39_event_universe() -> None:
    cases = _cases()
    events = [event for case in cases for event in case["events"]]

    assert len(cases) == 21
    assert len(events) == 39
    assert {
        metric: sum(event["metric"] == metric for event in events)
        for metric in MODULE.METRICS
    } == {"collision": 12, "oob": 13, "support": 14}
    assert sum(event["semantic_label"] == "invalid" for event in events) == 24
    assert sum(event["semantic_label"] == "ambiguous" for event in events) == 15
    assert sum(event["scoring_eligible"] for event in events) == 24
    assert all(
        not event["scoring_eligible"]
        for event in events
        if event["semantic_label"] == "ambiguous"
    )
    assert all(Path(case["blend_file"]).is_file() for case in cases)
    assert len({case["blend_file"] for case in cases}) == 21


def test_plan_is_judge_free_and_freezes_current_visual_contract() -> None:
    cases = _cases()
    selector_config = json.loads(
        (
            ROOT
            / "configs"
            / "models"
            / "gpt5_6_sol_litellm_local_visual_config_judge.json"
        ).read_text(encoding="utf-8")
    )
    plan = MODULE.build_plan(
        args=_args(),
        dataset_root=ROOT / "Support" / "datasets" / "cal_dataset1",
        materialized_roots=list(MODULE.DEFAULT_MATERIALIZED_ROOTS),
        cases=cases,
        selector_config=selector_config,
    )

    assert plan["counts"]["cases"] == 21
    assert plan["counts"]["events"] == 39
    assert plan["counts"]["event_arm_runs"] == 156
    assert plan["counts"]["events_by_split"] == {
        "obvious_distortion": 12,
        "subtle_distortion": 12,
        "fine_edge": 15,
    }
    assert plan["final_metric_judge"] == "disabled"
    assert plan["metric_verdicts_produced"] is False
    assert plan["accuracy_claim_supported"] is False
    assert plan["frozen"]["candidate_policy"] == "local"
    assert plan["frozen"]["candidate_count"] == 5
    assert plan["frozen"]["visual_config"] == MODULE.DEFAULT_P0B_VISUAL_CONFIGS
    assert tuple(plan["arms"]) == MODULE.ARMS


def test_selector_candidate_bank_cannot_be_silently_truncated() -> None:
    config = {"max_images": 5}
    MODULE.validate_selector_capacity(
        arms=("static_vlm_topk",),
        candidate_count=5,
        selector_config=config,
        allow_missing=False,
    )
    with pytest.raises(ValueError, match="exceeds selector max_images"):
        MODULE.validate_selector_capacity(
            arms=("static_vlm_topk",),
            candidate_count=6,
            selector_config=config,
            allow_missing=False,
        )
    with pytest.raises(ValueError, match="positive max_images"):
        MODULE.validate_selector_capacity(
            arms=("bounded_query_cov_all",),
            candidate_count=5,
            selector_config={},
            allow_missing=False,
        )


def test_unconditional_arm_uses_explicit_non_verdict_repair_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = {
        "metric": "collision",
        "event": {"object_a": "a", "object_b": "b"},
        "object_ids": ["a", "b"],
    }
    controlled = MODULE._provider_request_for_arm(
        "bounded_query_cov_all",
        request,
    )
    unchanged = MODULE._provider_request_for_arm(
        "static_vlm_topk",
        request,
    )

    assert unchanged == request
    assert "_camera_selection_phase" not in request
    assert controlled["_camera_selection_phase"] == "active_fallback"
    deficiency = controlled["_camera_evidence_deficiency"]
    assert deficiency["experimental_ablation"] is True
    assert deficiency["metric_verdict"] is None
    assert deficiency["camera_repairable"] is True
    assert deficiency["deficiencies"] == [
        {
            "code": "measured_local_visibility_insufficient",
            "repairability": "camera",
            "basis": "experimental_initial_repair_control",
        }
    ]

    captured: list[dict] = []

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(MODULE, "CameraEvidenceProvider", FakeProvider)
    MODULE._provider_for_arm(
        arm="bounded_query_cov_all",
        metric="collision",
        renderer=object(),
        selector=object(),
        blend_file=tmp_path / "scene.blend",
        out_dir=tmp_path / "evidence",
        candidate_count=5,
        max_steps=2,
        collision_geometry=None,
    )

    assert captured[0]["mode"] == "query_cov"
    assert captured[0]["active_repair"] is True
    assert captured[0]["max_steps"] == 2


def test_content_addressed_evidence_root_excludes_stale_manifests(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name in ("generated_scene.json", "event_gt.json"):
        (fixture / name).write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    geometry = tmp_path / "geometry.json"
    blend = tmp_path / "scene.blend"
    selector = tmp_path / "selector.json"
    for path, content in (
        (report, "{}"),
        (geometry, "{}"),
        (blend, "blend"),
        (selector, '{"max_images":5}'),
    ):
        path.write_text(content, encoding="utf-8")
    args = _args()
    args.selector_config = str(selector)
    case = {
        "case_id": "case",
        "fixture": str(fixture),
        "blend_file": str(blend),
        "report_path": str(report),
        "collision_geometry_path": str(geometry),
    }
    event = {
        "metric": "collision",
        "event_id": "a|b",
    }
    request = {"metric": "collision", "object_ids": ["a", "b"]}
    deterministic_key = MODULE._evidence_invocation_key(
        args=args,
        case=case,
        event=event,
        arm="deterministic_current",
        provider_request=request,
    )
    bounded_request = MODULE._provider_request_for_arm(
        "bounded_query_cov_all",
        request,
    )
    bounded_key = MODULE._evidence_invocation_key(
        args=args,
        case=case,
        event=event,
        arm="bounded_query_cov_all",
        provider_request=bounded_request,
    )
    assert deterministic_key != bounded_key

    root = tmp_path / "evidence_invocations"
    stale = root / "stale" / "event" / "camera_evidence_manifest.json"
    current = (
        root
        / bounded_key
        / "event"
        / "camera_evidence_manifest.json"
    )
    for path, marker in ((stale, "stale"), (current, "current")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"marker": marker}),
            encoding="utf-8",
        )

    artifacts = MODULE._artifact_records(
        root / bounded_key,
        "camera_evidence_manifest.json",
    )
    assert len(artifacts) == 1
    assert artifacts[0]["payload"]["marker"] == "current"


def test_aggregate_reports_sufficiency_and_cost_without_accuracy(
    tmp_path: Path,
) -> None:
    records = [
        {
            "case_id": "obvious_case",
            "split": "obvious_distortion",
            "metric": "collision",
            "event_id": "a|b",
            "semantic_label": "invalid",
            "scoring_eligible": True,
            "arm": "deterministic_current",
            "complete": True,
            "evidence_sufficiency": {
                "status": "sufficient",
                "repairability": None,
                "trigger_recommended": False,
            },
            "cost": {
                "active_used": False,
                "active_attempted": False,
                "selector_calls": 0,
                "camera_actions": 0,
                "elapsed_seconds": 1.5,
            },
            "shadow": {
                "active_attempted": False,
                "counterfactual_would_replace": False,
                "repair_success": False,
                "deterministic_before_assessment": {
                    "status": "sufficient",
                },
                "counterfactual_after_assessment": None,
                "trigger_reason_codes": [],
                "official_packet_source": "deterministic",
            },
        },
        {
            "case_id": "fine_case",
            "split": "fine_edge",
            "metric": "support",
            "event_id": "x",
            "semantic_label": "ambiguous",
            "scoring_eligible": False,
            "arm": "conditional_active_shadow",
            "complete": True,
            "evidence_sufficiency": {
                "status": "insufficient",
                "repairability": "camera",
                "trigger_recommended": True,
            },
            "cost": {
                "active_used": False,
                "active_attempted": True,
                "selector_calls": 2,
                "camera_actions": 1,
                "elapsed_seconds": 3.5,
            },
            "shadow": {
                "active_attempted": True,
                "counterfactual_would_replace": True,
                "repair_success": True,
                "deterministic_before_assessment": {
                    "status": "insufficient",
                    "reason_codes": ["support_focus_not_visible"],
                },
                "counterfactual_after_assessment": {
                    "status": "sufficient",
                },
                "trigger_reason_codes": ["support_focus_not_visible"],
                "active_error": None,
                "official_packet_source": "deterministic",
            },
        },
        {
            "case_id": "unknown_case",
            "split": "fine_edge",
            "metric": "oob",
            "event_id": "y",
            "semantic_label": "ambiguous",
            "scoring_eligible": False,
            "arm": "static_vlm_topk",
            "complete": True,
            "evidence_sufficiency": {
                "status": "unknown",
                "repairability": "unknown",
                "trigger_recommended": False,
            },
            "cost": {
                "active_used": False,
                "active_attempted": False,
                "selector_calls": 1,
                "camera_actions": 0,
                "elapsed_seconds": 2.0,
            },
            "shadow": {
                "active_attempted": False,
                "counterfactual_would_replace": False,
                "repair_success": False,
                "deterministic_before_assessment": None,
                "counterfactual_after_assessment": None,
                "trigger_reason_codes": [],
                "official_packet_source": "static_vlm_selection",
            },
        },
    ]
    for index, record in enumerate(records):
        path = (
            tmp_path
            / "cases"
            / record["case_id"]
            / "events"
            / f"event_{index}"
            / "arms"
            / record["arm"]
            / "result.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    summary = MODULE.aggregate_results(
        tmp_path,
        expected_plan={
            "counts": {"event_arm_runs": 3},
            "plan_sha256": "plan",
        },
    )

    assert summary["counts"] == {
        "results": 3,
        "expected_results": 3,
        "missing_results": 0,
        "complete": 3,
        "failures": 0,
        "unresolved": 2,
        "shadow_counterfactual_failures": 0,
        "scoring_eligible": 1,
        "ambiguous_non_scoring": 2,
    }
    assert summary["final_metric_judge"] == "disabled"
    assert summary["metric_verdicts_produced"] is False
    assert summary["accuracy_claim_supported"] is False
    conditional = next(
        item
        for item in summary["arms"]
        if item["arm"] == "conditional_active_shadow"
    )
    assert conditional["trigger_count"] == 1
    assert conditional["repair_success_count"] == 1
    assert conditional["repair_success_rate_on_triggered_subset"] == 1.0
    assert conditional["before_status_counts"] == {"insufficient": 1}
    assert conditional["after_status_counts"] == {"sufficient": 1}
    assert conditional["trigger_reason_counts"] == {
        "support_focus_not_visible": 1
    }
    assert conditional["official_packet_source_counts"] == {
        "deterministic": 1
    }
    assert conditional["selector_calls"] == 2
    assert conditional["camera_actions"] == 1
    assert conditional["metrics"]["support"]["trigger_count"] == 1
    static = next(
        item
        for item in summary["arms"]
        if item["arm"] == "static_vlm_topk"
    )
    assert static["official_status_counts"] == {"unknown": 1}
    assert static["unresolved_count"] == 1
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.tsv").is_file()
    assert (tmp_path / "failure_attribution.tsv").is_file()


def test_v2_trajectory_counts_and_shadow_semantics_are_normalized() -> None:
    camera_manifests = [
        {
            "path": "/tmp/camera.json",
            "payload": {
                "selection": {
                    "selector_call_count": 3,
                    "camera_action_count": 2,
                    "selected_view_ids": ["view"],
                    "stop_reason": "camera_action_budget_exhausted",
                    "steps": [
                        {
                            "action_execution": {"executed": True},
                            "decision": {"action": {"proposal_id": "p0"}},
                        },
                        {
                            "action_execution": {"executed": False},
                            "decision": {"action": {"proposal_id": "p1"}},
                        },
                    ],
                }
            },
        }
    ]
    fallback = [
        {
            "path": "/tmp/fallback.json",
            "payload": {
                "active_used": False,
                "active_attempted": True,
                "shadow_mode": True,
                "counterfactual_would_replace": True,
                "official_packet_source": "deterministic",
                "deterministic_assessment": {
                    "status": "insufficient",
                    "reason_codes": ["required_entities_not_jointly_visible"],
                },
                "final_assessment": {"status": "sufficient"},
                "active_error": None,
                "deterministic_error": None,
            },
        }
    ]

    cost = MODULE._cost_summary(
        camera_manifests,
        fallback,
        elapsed_seconds=5.0,
    )
    shadow = MODULE._shadow_summary(
        arm="conditional_active_shadow",
        official_assessment={"status": "insufficient"},
        fallback_manifests=fallback,
    )

    assert cost["selector_calls"] == 3
    assert cost["camera_actions"] == 2
    assert cost["active_attempted"] is True
    assert shadow["active_attempted"] is True
    assert shadow["counterfactual_would_replace"] is True
    assert shadow["repair_success"] is True
    assert shadow["deterministic_before_assessment"]["status"] == "insufficient"
    assert shadow["counterfactual_after_assessment"]["status"] == "sufficient"
    assert shadow["official_packet_source"] == "deterministic"

    not_triggered = MODULE._shadow_summary(
        arm="conditional_active_shadow",
        official_assessment={"status": "sufficient"},
        fallback_manifests=[
            {
                "path": "/tmp/not-triggered.json",
                "payload": {
                    "active_attempted": False,
                    "counterfactual_would_replace": True,
                    "official_packet_source": "deterministic",
                    "deterministic_assessment": {"status": "sufficient"},
                    "final_assessment": {"status": "sufficient"},
                },
            }
        ],
    )
    assert not_triggered["counterfactual_would_replace"] is False
    assert not_triggered["counterfactual_after_assessment"] is None


def test_run_manifest_records_completion_failures_and_result_hashes(
    tmp_path: Path,
) -> None:
    plan = {
        "plan_sha256": "plan-hash",
        "counts": {"event_arm_runs": 3},
    }
    (tmp_path / "experiment_plan.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    values = [
        {"complete": True},
        {"complete": False, "error": "EndpointHTTPError"},
    ]
    for index, value in enumerate(values):
        path = (
            tmp_path
            / "cases"
            / f"case_{index}"
            / "events"
            / "event"
            / "arms"
            / "arm"
            / "result.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    manifest = MODULE.build_run_manifest(
        out_dir=tmp_path,
        plan=plan,
        experiment_id="test",
        elapsed_seconds=7.0,
    )

    assert manifest["result_file_count"] == 2
    assert manifest["complete_result_count"] == 1
    assert manifest["failed_result_count"] == 1
    assert manifest["missing_result_count"] == 1
    assert len(manifest["result_sha256"]) == 2
    assert manifest["result_index_sha256"] == MODULE._json_sha256(
        manifest["result_sha256"]
    )
