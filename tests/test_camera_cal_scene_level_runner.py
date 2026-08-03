from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.evaluator.profile import L1, L2, L3
from benchmark.visual_judge.contracts import ResponseSchemaRepairError
from scripts import run_camera_cal_scene_level as runner


def test_default_discovery_covers_all_ready_camera_cal_cases() -> None:
    cases = runner.discover_cases(runner.DEFAULT_DATASET_ROOT)

    assert len(cases) == 30
    assert cases[0]["case_id"] == "N001"
    assert cases[-1]["case_id"] == "N030"


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
    assert style["model"]["prediction"] == "unresolved"
    assert style["included_in_accuracy"] is False


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
    assert len(created_providers) == 2
    assert created_providers[0] is captured["p0b_local_view_provider"]
    assert created_providers[1] is captured["l3_initial_evidence_provider"]
    assert created_providers[0].kwargs["out_dir"].name == "l1_camera"
    assert created_providers[1].kwargs["out_dir"].name == "l3_initial_camera"
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
