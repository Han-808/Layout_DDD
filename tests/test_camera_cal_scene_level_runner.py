from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.evaluator.profile import L1, L2, L3
from benchmark.models import EndpointConfigurationError
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from scripts import run_camera_cal_scene_level as runner


def _write_ready_discovery_case(
    root: Path,
    directory_id: str,
    *,
    manifest_id: str | None = None,
) -> None:
    case_root = root / directory_id
    (case_root / "scene").mkdir(parents=True)
    (case_root / "prepared").mkdir()
    (case_root / "evidence").mkdir()
    for path in (
        case_root / "scene/canonical_scene.json",
        case_root / "prepared/evaluation.blend",
        case_root / "annotation.json",
        case_root / "evidence/standardized_perspective.png",
        case_root / "evidence/standardized_top.png",
        case_root / "evidence/standardized_identity_map.png",
        case_root / "evidence/collision_geometry_manifest.json",
    ):
        path.write_bytes(b"{}")
    (case_root / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": manifest_id or directory_id,
                "status": "ready",
                "scene_type": "test",
                "object_count": 1,
                "paths": {},
            }
        )
    )


def test_default_discovery_covers_all_ready_camera_cal_cases() -> None:
    cases = runner.discover_cases(runner.DEFAULT_DATASET_ROOT)

    assert len(cases) == 30
    assert cases[0]["case_id"] == "N001"
    assert cases[-1]["case_id"] == "N030"


def test_discovery_accepts_s_namespace_cases(tmp_path: Path) -> None:
    case_root = tmp_path / "S061"
    (case_root / "scene").mkdir(parents=True)
    (case_root / "prepared").mkdir()
    (case_root / "evidence").mkdir()
    for path in (
        case_root / "scene/canonical_scene.json",
        case_root / "prepared/evaluation.blend",
        case_root / "annotation.json",
        case_root / "evidence/standardized_perspective.png",
        case_root / "evidence/standardized_top.png",
        case_root / "evidence/standardized_identity_map.png",
        case_root / "evidence/collision_geometry_manifest.json",
    ):
        path.write_bytes(b"{}")
    (case_root / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "S061",
                "status": "ready",
                "scene_type": "test",
                "object_count": 1,
                "paths": {},
            }
        )
    )

    cases = runner.discover_cases(tmp_path, case_ids=["S061"])

    assert [case["case_id"] for case in cases] == ["S061"]


def test_discovery_accepts_flexible_path_safe_case_ids(tmp_path: Path) -> None:
    case_id = "mr.api2-kimi-k3.layout_01.room_01"
    _write_ready_discovery_case(tmp_path, case_id)

    discovered = runner.discover_cases(tmp_path)
    selected = runner.discover_cases(tmp_path, case_ids=[case_id])

    assert [case["case_id"] for case in discovered] == [case_id]
    assert [case["case_id"] for case in selected] == [case_id]


@pytest.mark.parametrize(
    "case_id",
    (
        "",
        ".",
        "..",
        "../escape",
        "/absolute/escape",
        "nested/escape",
        r"nested\escape",
        r"C:\absolute\escape",
        "x" * 129,
    ),
)
def test_discovery_rejects_unsafe_selected_case_ids(
    tmp_path: Path, case_id: str
) -> None:
    with pytest.raises(ValueError, match="case ID"):
        runner.discover_cases(tmp_path, case_ids=[case_id])


def test_discovery_rejects_duplicate_selected_case_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        runner.discover_cases(tmp_path, case_ids=["S001", "S001"])


def test_discovery_rejects_ready_manifest_directory_alias(
    tmp_path: Path,
) -> None:
    _write_ready_discovery_case(
        tmp_path,
        "source-a",
        manifest_id="mr.duplicate",
    )

    with pytest.raises(ValueError, match="differs from its directory"):
        runner.discover_cases(tmp_path)


def test_promptless_profile_and_request_disable_l2_prompt_use() -> None:
    scene = {
        "request_id": "request-1",
        "scene_type": "living room",
    }
    request = runner.promptless_scene_request(
        scene,
        {"scene_type": "living room"},
    )
    profile = runner.promptless_l1_l3_profile()

    assert request["instruction"] == ""
    assert request["metadata"]["generation_prompt_withheld_from_evaluator"] is True
    assert profile[L1]["enabled"] is True
    assert profile[L2]["enabled"] is False
    assert all(
        metric["enabled"] is False and metric["weight"] == 0.0
        for metric in profile[L2]["metrics"].values()
    )
    assert profile[L3]["enabled"] is True


def test_l3_only_recovery_profile_disables_l1() -> None:
    profile = runner.promptless_l3_only_profile()
    args = runner.parse_args(["--output-root", "/tmp/l3-only", "--l3-only"])

    assert args.l3_only is True
    assert profile["layer_weights"] == {
        runner.L1: 0.0,
        runner.L2: 0.0,
        runner.L3: 1.0,
        runner.L4: 0.0,
    }
    assert profile[L1]["enabled"] is False
    assert all(
        metric["enabled"] is False and metric["weight"] == 0.0
        for metric in profile[L1]["metrics"].values()
    )
    assert profile[L3]["enabled"] is True


def test_audit_graph_export_is_explicitly_opt_in(tmp_path: Path) -> None:
    default_args = runner.parse_args(
        ["--output-root", str(tmp_path / "default")]
    )
    enabled_args = runner.parse_args(
        [
            "--output-root",
            str(tmp_path / "enabled"),
            "--export-audit-graphs",
        ]
    )

    assert default_args.export_audit_graphs is False
    assert enabled_args.export_audit_graphs is True


def test_endpoint_stability_preflight_defaults_to_ten_real_image_calls(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(["--output-root", str(tmp_path / "run")])

    assert args.endpoint_preflight_attempts == 10
    assert args.endpoint_preflight_timeout_seconds == 300


def test_deduction_multiplier_defaults_to_two_and_is_configurable(
    tmp_path: Path,
) -> None:
    default_args = runner.parse_args(
        ["--output-root", str(tmp_path / "default")]
    )
    unscaled_args = runner.parse_args(
        [
            "--output-root",
            str(tmp_path / "unscaled"),
            "--deduction-multiplier",
            "1.0",
        ]
    )

    assert default_args.deduction_multiplier == 2.0
    assert unscaled_args.deduction_multiplier == 1.0
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--output-root",
                str(tmp_path / "invalid"),
                "--deduction-multiplier",
                "0",
            ]
        )


def test_functional_group_local_granularity_supports_both_modes(
    tmp_path: Path,
) -> None:
    default_args = runner.parse_args(
        ["--output-root", str(tmp_path / "default")]
    )
    batched_args = runner.parse_args(
        [
            "--output-root",
            str(tmp_path / "batched"),
            "--functional-group-local-granularity",
            "batched",
        ]
    )
    shared_args = runner.parse_args(
        [
            "--output-root",
            str(tmp_path / "shared"),
            "--functional-group-local-evidence-policy",
            "shared_group_bank",
        ]
    )

    assert default_args.functional_group_local_granularity == "per_check"
    assert (
        default_args.functional_group_local_evidence_policy
        == "shared_group_bank"
    )
    assert batched_args.functional_group_local_granularity == "batched"
    assert (
        batched_args.functional_group_local_evidence_policy
        == "shared_group_bank"
    )
    assert (
        shared_args.functional_group_local_evidence_policy
        == "shared_group_bank"
    )
    assert runner.scene_quality_config(
        ("functional_consistency",),
        functional_group_local_granularity="per_check",
    )["metrics"]["functional_consistency"][
        "group_local_check_granularity"
    ] == "per_check"
    assert runner.scene_quality_config(
        ("functional_consistency",),
        functional_group_local_granularity="batched",
        functional_group_local_evidence_policy="isolated_episode",
    )["metrics"]["functional_consistency"][
        "group_local_check_granularity"
    ] == "batched"
    shared = runner.scene_quality_config(
        ("functional_consistency",),
        functional_group_local_granularity="per_check",
        functional_group_local_evidence_policy="shared_group_bank",
    )["metrics"]["functional_consistency"]
    assert shared["group_local_evidence_policy"] == "shared_group_bank"
    assert shared["group_local_active_window_max_images"] == 6
    with pytest.raises(ValueError, match="requires.*per_check"):
        runner.scene_quality_config(
            ("functional_consistency",),
            functional_group_local_granularity="batched",
            functional_group_local_evidence_policy="shared_group_bank",
        )


def test_parallel_fail_fast_records_running_failures_and_cancellations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress = runner.ProgressReporter(
        tmp_path / "progress.jsonl",
        terminal=False,
    )
    cases = [{"case_id": f"N{index:03d}"} for index in range(1, 9)]
    first_wave = threading.Barrier(2)

    def fake_run_case(*, case: dict[str, Any], **_: Any) -> dict[str, Any]:
        first_wave.wait(timeout=2.0)
        if case["case_id"] != "N001":
            time.sleep(0.1)
        raise ValueError(f"failed:{case['case_id']}")

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    records, failures = runner.run_cases_parallel(
        cases=cases,
        case_kwargs={},
        output_root=tmp_path,
        progress=progress,
        max_workers=2,
        continue_on_error=False,
    )

    assert {record["case_id"] for record in records} == {
        case["case_id"] for case in cases
    }
    assert failures
    assert any(record["status"] == "cancelled" for record in records)
    assert all(record["status"] in {"failed", "cancelled"} for record in records)
    for record in records:
        manifest = runner.read_json(
            tmp_path
            / "cases"
            / record["case_id"]
            / "case_run_manifest.json"
        )
        assert manifest["status"] == record["status"]


def test_route_is_explicit_and_never_falls_back_to_port_4000() -> None:
    with pytest.raises(RuntimeError, match="explicit runtime model routing"):
        runner.effective_model_route({})

    stale = {
        "JUDGE_ENDPOINT": "http://127.0.0.1:4000/v1",
        "JUDGE_MODEL": "model",
        "JUDGE_API_KEY_ENV": "TEST_KEY",
        "TEST_KEY": "secret",
    }
    with pytest.raises(RuntimeError, match="stale LiteLLM route"):
        runner.effective_model_route(stale)

    current = {
        "JUDGE_ENDPOINT": "http://127.0.0.1:4010/v1",
        "JUDGE_MODEL": "model",
        "JUDGE_API_KEY_ENV": "TEST_KEY",
        "TEST_KEY": "secret",
    }
    route = runner.effective_model_route(current)
    assert route == {
        "endpoint": "http://127.0.0.1:4010/v1",
        "model": "model",
        "api_key_env": "TEST_KEY",
        "authorization_configured": True,
    }
    assert "secret" not in json.dumps(runner.safe_route_manifest(route))


def test_grouping_model_uses_dedicated_completion_budget() -> None:
    model = runner.build_grouping_model(
        {
            "endpoint": "http://127.0.0.1:4010/v1",
            "model": "gpt-5.6-sol",
            "api_key_env": "LITELLM_MASTER_KEY",
        }
    )

    assert runner.GROUPING_COMPLETION_MAX_TOKENS == 3192
    assert model.max_tokens == 3192


def test_scene_comparison_keeps_unclear_and_unresolved_explicit() -> None:
    comparison = runner.build_scene_comparison(
        case_id="N001",
        annotation={
            "metrics": {
                "scale_consistency": {
                    "anomaly": True,
                    "unclear": True,
                    "affected_object_ids": ["chair"],
                },
                "style_consistency": {
                    "anomaly": False,
                    "unclear": False,
                    "affected_object_ids": [],
                },
            }
        },
        scene_quality_report={
            "metrics": {
                "scale_consistency": {
                    "status": "evaluated",
                    "score": 0.0,
                    "final_object_findings": [
                        {"object_id": "floor_lamp"}
                    ],
                    "coverage": {
                        "eligible_count": 2,
                        "resolved_count": 2,
                    },
                },
                "style_consistency": {
                    "status": "unresolved",
                    "score": None,
                    "coverage": {
                        "eligible_count": 1,
                        "resolved_count": 0,
                    },
                },
            }
        },
        metrics=("scale_consistency", "style_consistency"),
    )

    scale = comparison["metrics"]["scale_consistency"]
    style = comparison["metrics"]["style_consistency"]
    assert scale["model"]["prediction"] == "invalid"
    assert scale["included_in_accuracy"] is False
    assert scale["matches"] is None
    assert scale["anomaly_level"] == {
        "scope": "anomaly_object_attribution",
        "included_in_accuracy": False,
        "human_object_ids": ["chair"],
        "model_object_ids": ["floor_lamp"],
        "true_positive_object_ids": [],
        "false_negative_object_ids": ["chair"],
        "false_positive_object_ids": ["floor_lamp"],
        "precision": None,
        "recall": None,
        "exact_match": None,
        "covered_any_human_anomaly": None,
        "exclusion_reason": "human_annotation_unclear",
    }
    assert style["model"]["prediction"] == "unresolved"
    assert style["included_in_accuracy"] is False


def test_metric_prediction_uses_verdict_not_posthoc_burden_score() -> None:
    assert runner.metric_prediction(
        {
            "status": "evaluated",
            "score": 0.8,
            "verdict_score": 0.0,
            "judgement": {"verdict": "invalid"},
        }
    ) == "invalid"
    assert runner.metric_prediction(
        {
            "status": "evaluated",
            "score": 0.8,
            "verdict_score": 1.0,
            "judgement": {"verdict": "valid"},
        }
    ) == "valid"


def test_metric_prediction_separates_infrastructure_failure_from_unresolved() -> None:
    assert runner.metric_prediction(
        {
            "status": "failed",
            "terminal_state": "infrastructure_failure",
            "score": None,
        }
    ) == "infrastructure_failure"
    assert runner.metric_prediction(
        {"status": "not_applicable", "score": None}
    ) == "unresolved"


def test_scene_match_does_not_hide_anomaly_object_attribution_failure():
    comparison = runner.build_scene_comparison(
        case_id="N020",
        annotation={
            "metrics": {
                "functional_consistency": {
                    "anomaly": True,
                    "unclear": False,
                    "affected_object_ids": [
                        "lounge_sofa",
                        "bookshelf_01",
                        "bookshelf_02",
                        "display_cabinet",
                    ],
                }
            }
        },
        scene_quality_report={
            "metrics": {
                "functional_consistency": {
                    "status": "evaluated",
                    "score": 0.0,
                    "final_object_findings": [
                        {"object_id": "coffee_machine"},
                        {"object_id": "side_table"},
                    ],
                }
            }
        },
        metrics=("functional_consistency",),
    )

    metric = comparison["metrics"]["functional_consistency"]
    assert metric["matches"] is True
    assert metric["anomaly_level"]["exact_match"] is False
    assert metric["anomaly_level"][
        "false_negative_object_ids"
    ] == [
        "lounge_sofa",
        "bookshelf_01",
        "bookshelf_02",
        "display_cabinet",
    ]
    assert metric["anomaly_level"][
        "false_positive_object_ids"
    ] == ["coffee_machine", "side_table"]
    assert comparison["comparison_scopes"] == [
        "scene_level_metric_verdict",
        "anomaly_object_attribution",
    ]


def test_l3_resolution_audit_accepts_publishable_partial_coverage() -> None:
    audit = runner.l3_resolution_audit(
        {
            "metrics": {
                "functional_consistency": {
                    "status": "evaluated",
                    "score": 0.8,
                    "coverage": {
                        "complete": False,
                        "score_grounding": {
                            "fraction": 0.8,
                            "complete": False,
                        },
                        "score_projection": {
                            "coverage_threshold_passed": True,
                        },
                    },
                    "functional_check_coverage": {
                        "complete": False,
                        "unresolved_check_ids": ["functional_check_001"],
                    },
                },
                "semantic_placement_consistency": {
                    "status": "evaluated",
                    "score": 1.0,
                    "coverage": {"complete": True},
                },
            }
        },
        metrics=(
            "functional_consistency",
            "semantic_placement_consistency",
        ),
    )

    assert audit["status"] == "resolved"
    assert audit["unresolved_metrics"] == []
    assert audit["infrastructure_failure_metrics"] == []
    assert audit["partial_coverage_metrics"] == [
        "functional_consistency"
    ]
    assert audit["coverage_warnings_by_metric"][
        "functional_consistency"
    ] == [
        "coverage:incomplete",
        "functional_check_coverage:incomplete",
    ]


def test_l3_resolution_audit_keeps_below_threshold_out_of_infra() -> None:
    audit = runner.l3_resolution_audit(
        {
            "metrics": {
                "functional_consistency": {
                    "status": "evaluated",
                    "score": None,
                    "coverage": {
                        "complete": False,
                        "score_grounding": {
                            "fraction": 0.7,
                            "complete": False,
                        },
                        "score_projection": {
                            "coverage_threshold_passed": False,
                        },
                    },
                    "functional_check_coverage": {"complete": False},
                }
            }
        },
        metrics=("functional_consistency",),
    )

    assert audit["status"] == "unresolved"
    assert audit["unresolved_metrics"] == ["functional_consistency"]
    assert audit["infrastructure_failure_metrics"] == []
    assert audit["below_coverage_threshold_metrics"] == [
        "functional_consistency"
    ]
    assert audit["reasons_by_metric"]["functional_consistency"] == [
        "coverage:incomplete",
        "functional_check_coverage:incomplete",
        "coverage:below_publishable_threshold",
    ]


def test_l3_resolution_audit_preserves_explicit_infrastructure_failure() -> None:
    audit = runner.l3_resolution_audit(
        {
            "metrics": {
                "functional_consistency": {
                    "status": "failed",
                    "terminal_state": "infrastructure_failure",
                    "score": None,
                    "infrastructure_failures": [
                        {"failure_kind": "engineering_failure"}
                    ],
                }
            }
        },
        metrics=("functional_consistency",),
    )

    assert audit["status"] == "infrastructure_failure"
    assert audit["unresolved_metrics"] == []
    assert audit["infrastructure_failure_metrics"] == [
        "functional_consistency"
    ]


def test_case_resume_requires_exact_fingerprint_and_complete_outputs(
    tmp_path: Path,
) -> None:
    case_out = tmp_path / "N001"
    case_out.mkdir()
    for name in (
        "evaluation_report.json",
        "grouping.json",
        "l1_report.json",
        "l1_diagnostics.json",
        "scene_quality_report.json",
        "scene_comparison.json",
        "control_manifest.json",
    ):
        (case_out / name).write_text("{}\n", encoding="utf-8")

    assert runner.resumable_case(
        {
            "status": "complete",
            "input_fingerprint": "expected",
        },
        expected_fingerprint="expected",
        case_out=case_out,
    )
    assert not runner.resumable_case(
        {
            "status": "complete",
            "input_fingerprint": "stale",
        },
        expected_fingerprint="expected",
        case_out=case_out,
    )


def test_run_case_keeps_l1_scene_provider_separate_from_l3_group_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_case = runner.discover_cases(
        runner.DEFAULT_DATASET_ROOT,
        case_ids=["N001"],
    )[0]
    created_providers: list[Any] = []
    captured: dict[str, Any] = {}

    class FakeProvider:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.candidate_policy = "local_v1"
            created_providers.append(self)

    monkeypatch.setattr(runner, "CameraEvidenceProvider", FakeProvider)
    monkeypatch.setattr(
        runner,
        "BlenderRenderer",
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )
    monkeypatch.setattr(
        runner,
        "DeterministicLocalCameraSelector",
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )
    monkeypatch.setattr(
        runner,
        "CameraViewEvidenceRenderer",
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )
    monkeypatch.setattr(
        runner,
        "CameraCandidatePreviewRenderer",
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )
    monkeypatch.setattr(
        runner,
        "build_openai_compatible_vlm_judge",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        runner,
        "build_openai_compatible_camera_selector",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        runner,
        "build_grouping_model",
        lambda route: SimpleNamespace(route=route),
    )
    monkeypatch.setattr(
        runner,
        "load_collision_geometry_manifest",
        lambda path: {"schema_version": "collision_geometry_v1"},
    )

    def fake_run_evaluate(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        metric_report = {
            "metric": "functional_consistency",
            "status": "evaluated",
            "score": 0.0,
            "reason": None,
            "coverage": {
                "eligible_count": 3,
                "resolved_count": 3,
            },
            "judge_call_count": 3,
            "group_results": [
                {
                    "group_id": "group_001",
                    "status": "evaluated",
                    "score": 0.0,
                    "evidence_paths": ["one.png"],
                    "evidence_resolution": {"provider_status": "complete"},
                }
            ],
        }
        report = {
            "benchmark_score_status": "insufficient_metric_coverage",
            "scoring_reliability": {
                "schema_version": "scoring_reliability_v2",
                "terminal_state": "complete",
            },
            "reports": {
                "object_grouping": {
                    "status": "complete",
                    "object_groups": [
                        {
                            "group_id": "group_001",
                            "object_ids": ["television", "tv_cabinet"],
                        }
                    ],
                },
                "scene_quality": {
                    "status": "not_applicable",
                    "metrics": {
                        "functional_consistency": metric_report,
                    },
                },
            },
            "layer_reports": {
                L1: {
                    "status": "evaluated",
                    "metrics": {
                        "collision": {"status": "evaluated", "score": 1.0},
                        "oob": {"status": "evaluated", "score": 1.0},
                        "support": {"status": "evaluated", "score": 1.0},
                    },
                }
            },
            "evaluation_config": {
                "vlm_evaluation_control": {
                    "integration": {
                        "runtime": {
                            "controlled_calls": [],
                        }
                    }
                }
            },
        }
        runner.atomic_write_json(Path(kwargs["out"]), report)
        return deepcopy(report)

    monkeypatch.setattr(runner, "run_evaluate", fake_run_evaluate)
    output_root = tmp_path / "run"
    record = runner.run_case(
        case=source_case,
        dataset_root=runner.DEFAULT_DATASET_ROOT,
        output_root=output_root,
        grouping_config_path=runner.DEFAULT_GROUPING_CONFIG,
        route={
            "endpoint": "http://127.0.0.1:4010/v1",
            "model": "model",
            "api_key_env": "TEST_KEY",
            "authorization_configured": True,
        },
        metrics=("functional_consistency",),
        renderer_config={
            "blender_bin": "/fake/blender",
            "timeout_seconds": 10,
            "width": 64,
            "height": 64,
            "render_engine": "BLENDER_EEVEE_NEXT",
            "cycles_device": "CPU",
            "cycles_samples": 1,
            "cycles_denoising": False,
            "preview_render_engine": "BLENDER_EEVEE_NEXT",
            "preview_width": 64,
            "preview_height": 64,
            "preview_cycles_samples": 1,
        },
        control_config=runner.resolved_control().to_dict(),
        resume=True,
    )

    assert record["status"] == "complete"
    assert len(created_providers) == 3
    assert (
        created_providers[0]
        is captured["p0b_local_view_provider"]._provider
    )
    assert (
        created_providers[1]
        is captured["l3_initial_evidence_provider"]._provider
    )
    assert (
        created_providers[2]
        is captured["functional_probe_evidence_provider"]._provider
    )
    assert created_providers[0].kwargs["out_dir"].name == "l1_camera"
    assert created_providers[1].kwargs["out_dir"].name == "l3_initial_camera"
    assert created_providers[2].kwargs["out_dir"].name == (
        "l3_functional_probes"
    )
    assert created_providers[2].kwargs["mode"] == "query_cov"
    assert created_providers[2].kwargs["max_views"] == 1
    assert created_providers[2].kwargs["max_steps"] == 0
    assert created_providers[2].kwargs["candidate_count"] == 6
    assert (
        captured["functional_evidence_planner"]
        is created_providers[2].kwargs["selector"]
    )
    assert captured["scene_request"]["instruction"] == ""
    assert captured["evaluation_profile"][L2]["enabled"] is False
    assert captured["p0b_official_mode"] is False
    assert captured["scene_quality_config"]["metrics"][
        "functional_consistency"
    ]["enabled"] is True
    assert captured["specification_contract"] is None
    assert record["final_decision_status"] == "resolved"
    diagnostics = runner.read_json(
        output_root / "cases" / "N001" / "l1_diagnostics.json"
    )
    assert diagnostics["engineering_failure_count"] == 0
    assert diagnostics["l3_diagnostics_completed"] is True
    assert record["api_usage"]["api_calls_number"] == 0
    manifest = runner.read_json(
        output_root / "cases" / "N001" / "case_run_manifest.json"
    )
    assert manifest["scoring_reliability"] == {
        "schema_version": "scoring_reliability_v2",
        "terminal_state": "complete",
    }
    assert (
        manifest["benchmark_score_status"]
        == "insufficient_metric_coverage"
    )
    assert (
        runner.read_json(
            output_root / "cases" / "N001" / "api_usage.json"
        )["token_usage_coverage"]
        == "not_applicable"
    )


def test_runner_uses_expanded_group_judge_evidence_budget() -> None:
    control = runner.resolved_control()
    judge_config = runner.model_config(
        {
            "endpoint": "http://127.0.0.1:4010/v1",
            "model": "test-model",
            "api_key_env": "TEST_KEY",
        },
        role="judge",
    )

    assert control.max_evidence_rounds == 3
    assert control.max_total_images == 8
    assert control.max_camera_actions == 3
    assert control.max_selector_calls == 4
    assert control.max_views_per_round == 2
    assert judge_config["max_images"] == 8
    assert runner.JUDGE_COMPLETION_MAX_TOKENS == 8192
    assert judge_config["max_tokens"] == 8192
    assert runner.model_config(
        {
            "endpoint": "http://127.0.0.1:4010/v1",
            "model": "test-model",
            "api_key_env": "TEST_KEY",
        },
        role="camera-selector",
    )["max_tokens"] == 2048


def test_api_tracker_persists_progress_calls_and_reported_tokens(
    tmp_path: Path,
) -> None:
    progress = runner.ProgressReporter(
        tmp_path / "progress.jsonl",
        terminal=False,
    )
    tracker = runner.APICallTracker(
        case_id="N001",
        calls_path=tmp_path / "api_calls.jsonl",
        usage_path=tmp_path / "api_usage.json",
        progress=progress,
    )

    class FakeModel:
        model_id = "fake-model"
        endpoint = "http://127.0.0.1:4010/v1"

        def __init__(self) -> None:
            self.last_request_metadata: dict[str, Any] = {}

        def chat_messages(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> str:
            self.last_request_metadata = {
                "endpoint": self.endpoint,
                "model": self.model_id,
                "message_count": len(messages),
                "image_count": 0,
                "prompt_chars": 12,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {
                        "cached_tokens": 2,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 1,
                    },
                },
            }
            return json.dumps(
                {
                    "ok": True,
                    "call_type": kwargs.get("call_type"),
                }
            )

    observed = tracker.observe_model(FakeModel(), role="judge")
    result = observed.chat_messages(
        [{"role": "user", "content": "hello"}],
        call_type="vlm_judge.canonical.style_consistency",
    )

    assert json.loads(result)["ok"] is True
    usage = runner.read_json(tmp_path / "api_usage.json")
    assert usage["api_calls_number"] == 1
    assert usage["token_usage_coverage"] == "complete"
    assert usage["tokens_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cached_prompt_tokens": 2,
        "reasoning_tokens": 1,
    }
    assert usage["by_role"]["judge"]["api_calls_number"] == 1
    calls = runner.read_api_call_records(
        tmp_path / "api_calls.jsonl"
    )
    assert calls[0]["call_type"] == (
        "vlm_judge.canonical.style_consistency"
    )
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events == ["api_call_started", "api_call_completed"]


def test_api_tracker_trips_shared_route_circuit_breaker(
    tmp_path: Path,
) -> None:
    progress = runner.ProgressReporter(
        tmp_path / "progress.jsonl",
        terminal=False,
    )
    signal = runner.ModelRouteAbortSignal()
    tracker = runner.APICallTracker(
        case_id="N001",
        calls_path=tmp_path / "api_calls.jsonl",
        usage_path=tmp_path / "api_usage.json",
        progress=progress,
        model_route_abort_signal=signal,
    )

    class BrokenModel:
        model_id = "claude-opus-5-aihub"
        endpoint = "http://127.0.0.1:4010/v1"
        last_request_metadata: dict[str, Any] = {}

        def chat_messages(self, messages: list[dict], **_: Any) -> str:
            del messages
            raise EndpointConfigurationError(
                "HTTP 400: on-demand throughput isn't supported; use an "
                "inference profile"
            )

    observed = tracker.observe_model(BrokenModel(), role="judge")
    with pytest.raises(EndpointConfigurationError):
        observed.chat_messages([{"role": "user", "content": "first"}])
    assert signal.is_set() is True
    assert tracker.summary()["api_calls_number"] == 1

    with pytest.raises(EndpointConfigurationError, match="route was disabled"):
        observed.chat_messages([{"role": "user", "content": "second"}])
    assert tracker.summary()["api_calls_number"] == 1


def test_api_usage_marks_missing_endpoint_usage_without_estimating() -> None:
    summary = runner.api_usage_summary(
        [
            {
                "role": "grouping",
                "call_type": "vlm_grouping.partition",
                "status": "complete",
                "tokens_usage": None,
            }
        ]
    )

    assert summary["api_calls_number"] == 1
    assert summary["token_usage_coverage"] == "unavailable"
    assert summary["tokens_usage"] is None
    assert summary["token_usage_estimated"] is False


def test_api_usage_separates_functional_discovery_surface_selector_and_judge(
) -> None:
    records = [
        {
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.affordance"
            ),
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.relations"
            ),
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": "vlm_camera_pose.usable_surface_decode",
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": "vlm_camera_pose.query_cov",
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": "camera_selector_candidate_only",
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "judge",
            "call_type": "vlm_judge.canonical.functional_consistency",
            "status": "complete",
            "tokens_usage": None,
        },
    ]

    summary = runner.api_usage_summary(records)

    assert summary["operation_calls"] == {
        "functional_discovery": 2,
        "functional_affordance": 1,
        "functional_relation": 1,
        "placement_discovery": 0,
        "usable_surface_decoder": 1,
        "camera_selector": 2,
        "judge": 1,
    }


def test_api_usage_counts_functional_schema_repair_in_call_family() -> None:
    records = [
        {
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.affordance"
            ),
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.affordance"
                ".schema_repair"
            ),
            "status": "complete",
            "tokens_usage": None,
        },
        {
            "role": "camera_selector",
            "call_type": (
                "vlm_camera_pose.functional_discovery.relations"
            ),
            "status": "complete",
            "tokens_usage": None,
        },
    ]

    summary = runner.api_usage_summary(records)

    assert summary["api_calls_number"] == 3
    assert summary["operation_calls"]["functional_discovery"] == 3
    assert summary["operation_calls"]["functional_affordance"] == 2
    assert summary["operation_calls"]["functional_relation"] == 1


def test_l1_schema_failure_marks_scene_unresolved_but_keeps_l3_diagnostic(
) -> None:
    schema_audit = {
        "policy": "single_schema_repair_retry_v1",
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": False,
        "attempts": [
            {
                "attempt": 1,
                "raw_response": '{"defects":["bad"]}',
            },
            {
                "attempt": 2,
                "raw_response": '{"defects":["bad"]}',
            },
        ],
    }
    l1_report = {
        "status": "unresolved",
        "metrics": {
            "collision": {
                "metric": "collision",
                "status": "requires_vlm",
                "pairs": [
                    {
                        "route": "vlm_adjudication_failed",
                        "adjudication_error": "response schema invalid",
                        "adjudication_failure_audit": schema_audit,
                    }
                ],
            }
        },
    }

    failures = runner.collect_l1_engineering_failures(l1_report)
    summary = runner.binary_schema_validation_summary(l1_report)

    assert len(failures) == 1
    assert failures[0]["metric"] == "collision"
    assert failures[0]["response_schema_validation"] == schema_audit
    assert summary == {
        "logical_binary_judge_calls": 1,
        "response_attempts": 2,
        "schema_repair_retries": 1,
        "schema_repair_recoveries": 0,
        "schema_repair_failures": 1,
    }


def test_record_case_failure_closes_running_manifest_and_saves_raw_audit(
    tmp_path: Path,
) -> None:
    case_out = tmp_path / "cases" / "N001"
    case_out.mkdir(parents=True)
    runner.atomic_write_json(
        case_out / "case_run_manifest.json",
        {
            "schema_version": runner.CASE_SCHEMA_VERSION,
            "case_id": "N001",
            "status": "running",
        },
    )
    schema_audit = {
        "policy": "single_schema_repair_retry_v1",
        "attempt_count": 2,
        "repair_retry_count": 1,
        "recovered": False,
        "attempts": [
            {"attempt": 1, "raw_response": '{"defects":"bad"}'},
            {"attempt": 2, "raw_response": '{"defects":"bad"}'},
        ],
    }
    error = ResponseSchemaRepairError(
        "response remained invalid",
        schema_audit=schema_audit,
    )

    failure = runner.record_case_failure(
        case={"case_id": "N001"},
        output_root=tmp_path,
        error=error,
    )

    manifest = runner.read_json(
        case_out / "case_run_manifest.json"
    )
    assert failure["response_schema_validation"] == schema_audit
    assert manifest["status"] == "failed"
    assert manifest["final_decision_status"] == "unresolved"
    assert manifest["binary_response_schema_validation"] == schema_audit
