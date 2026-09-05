"""Acceptance gates over real evaluator reports with external-only CI fixtures."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmark.evaluator.profile import L0, L1, L2, L3
from benchmark.generation_comparison.evaluation_acceptance import (
    COMPLETE_SCORE, EXEMPT_METRICS, FROZEN_OWNERSHIP, FROZEN_REQUIRED_METRICS,
    acceptance_policy_name, evaluate_report_acceptance,
)
from benchmark.generation_comparison.evaluation_runtime import (
    CanonicalEvaluationRuntime, evaluator_policy_readiness,
)
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.pilot import prepare_controlled_pilot, run_prepared_pilot
from benchmark.generation_comparison.protocol import ComparisonProtocol
from benchmark.utils.io import read_json, write_json
from test_controlled_generation_pilot import _asset_root, _pilot_spec
from test_pipeline_evaluation_runtime import _install_external_fixtures, _runtime_config


def _policy():
    return {"acceptance_policy": FROZEN_REQUIRED_METRICS,
            "static_kwargs": {"asset_policy": dict(FROZEN_OWNERSHIP)}}


def _state():
    return {"renders": [], "judgements": [], "grouping_calls": 0,
            "functional_discovery": 0, "local_requests": []}


@pytest.fixture(scope="module")
def accepted_pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("accepted_pilot")
    spec = _pilot_spec(methods=["catalog_placement"])
    # Keep both cases at the existing one-object external-observation fixture's
    # scope; this test exercises scheduler advancement, not multi-object judging.
    spec["cases"][1]["objects"] = deepcopy(spec["cases"][0]["objects"])
    spec["evaluator"].update(_policy())
    state = _state()
    with pytest.MonkeyPatch.context() as patch:
        _install_external_fixtures(patch, state)
        patch.setenv("PIPELINE_TEST_JUDGE_KEY", "test-only")
        prepared = prepare_controlled_pilot(
            spec=spec, asset_root=_asset_root(root / "assets"), out_dir=root / "pilot",
            evaluation_runtime_config=_runtime_config(), case_ids=["case_001", "case_002"],
        )
        outputs = {}
        for count, case in enumerate(prepared["cases"], start=1):
            outputs[case["case_id"]] = write_json(root / f"native_{count}.json", {
                "schema_version": "catalog_placement_v1", "instances": [{
                    "instance_id": f"chair_instance_{index}", "slot_id": f"chair_{index}",
                    "asset_id": "chair_asset", "center_m": [1 + index * 2, 2, 0.5],
                    "uniform_scale": 1.0, "rotation_euler_xyz_deg": [0, 0, 0],
                } for index in range(1)],
            })
        before = {path: path.read_bytes() for path in outputs.values()}
        result = run_prepared_pilot(
            prepared_dir=root / "pilot", allow_offline_artifacts=True,
            method_outputs={"catalog_placement": outputs},
        )
        assert all(path.read_bytes() == data for path, data in before.items())
    rows = [json.loads(line) for line in (root / "pilot/results.jsonl").read_text().splitlines()]
    return {"root": root, "prepared": prepared, "result": result, "rows": rows,
            "state": state, "report": read_json(rows[0]["evaluation_report"])}


def test_opt_in_advances_pilot_without_relabeling_raw_scores(accepted_pilot):
    fixture = accepted_pilot
    result, rows, report = fixture["result"], fixture["rows"], fixture["report"]
    assert result["status"] == "completed", rows
    assert result["attempted_runs"] == result["accepted_evaluations"] == 2
    assert result["complete_evaluations"] == 0
    assert result["dry_run_passed_methods"] == ["catalog_placement"]
    assert len(fixture["state"]["renders"]) == 2
    assert result["real_upstream_execution_performed"] is False
    for row in rows:
        assert row["evaluation_accepted"] and not row["evaluation_success"]
        assert row["benchmark_score_status"] == "partial_coverage"
        assert row["evaluation_acceptance"]["accepted_partial_coverage"]
        run_manifest = read_json(row["run_manifest"])
        assert run_manifest["evaluator"]["acceptance"] == row["evaluation_acceptance"]
        assert run_manifest["evaluator"]["actual_policy_sha256"] == fixture["prepared"]["evaluator_config_sha256"]
    summary = read_json(result["summary"])
    assert summary["methods"]["catalog_placement"]["accepted_partial_evaluations"] == 2
    assert summary["methods"]["catalog_placement"]["mean_score"] is None
    assert summary["paired_score_deltas"] == []
    assert report["benchmark_score_status"] == "partial_coverage"
    assert report["layer_reports"][L3]["coverage"]["fraction"] == pytest.approx(0.84)


def test_gate_is_pure_and_strict_default_is_unchanged(accepted_pilot):
    report = accepted_pilot["report"]
    before = deepcopy(report)
    assert not evaluate_report_acceptance(report, {})["accepted"]
    decision = evaluate_report_acceptance(report, _policy())
    assert decision["accepted"] and not decision["scoring_modified"]
    assert decision["report_sha256"] == canonical_json_sha256(report)
    assert decision["exempt_metric_ids"] == sorted(f"{L3}.{name}" for name in EXEMPT_METRICS)
    assert report == before
    assert evaluate_report_acceptance({"benchmark_score": 0, "benchmark_score_status": "complete"}, {})["accepted"]


@pytest.mark.parametrize("score", [0.0, 0.1, 1.0])
def test_quality_score_is_not_an_acceptance_cutoff(accepted_pilot, score):
    report = deepcopy(accepted_pilot["report"])
    report["benchmark_score"] = score
    assert evaluate_report_acceptance(report, _policy())["accepted"]


@pytest.mark.parametrize("fraction,accepted", [(0.7999, False), (0.8, True), (0.84, True),
                                             (None, False), ("0.9", False), (True, False), (1.1, False)])
def test_l3_threshold_is_inclusive_and_not_overall_coverage(accepted_pilot, fraction, accepted):
    report = deepcopy(accepted_pilot["report"])
    report["coverage"]["grounded_score_fraction"] = 1.0
    report["layer_reports"][L3]["coverage"]["fraction"] = fraction
    assert evaluate_report_acceptance(report, _policy())["accepted"] is accepted


@pytest.mark.parametrize("path,value", [
    (("benchmark_score",), None),
    (("benchmark_score",), True),
    (("benchmark_score_status",), "insufficient_metric_coverage"),
    (("report_schema_version",), "unknown"),
    (("layer_reports", L0, "status"), "failed"),
    (("coverage", "score_resolution_complete"), False),
    (("coverage", "coverage_threshold_passed"), False),
    (("coverage", "comparability_signature"), None),
    (("coverage", "active_layers"), []),
    (("coverage", "covered_layers"), [[L1], L3]),
    (("coverage", "resolved_metrics_by_layer", L3), 3),
    (("coverage", "active_metrics_by_layer", L3), ["scale_consistency", {}]),
    (("scoring_reliability", "schema_version"), None),
    (("scoring_reliability", "infrastructure_failures"), [{"error": "render_failed"}]),
    (("scoring_reliability", "unresolved_claims"), [{"claim_id": "unjudged"}]),
    (("scoring_reliability", "unresolved_metric_ids"), [f"{L3}.functional_consistency"]),
    (("scoring_reliability", "metrics"), []),
    (("evaluation_config", "asset_policy", "appearance_owner"), "generator"),
    (("evaluation_config", "metric_applicability", L3, "style_consistency", "applicability"), "unknown"),
    (("layer_reports", L1, "coverage", "complete"), False),
    (("layer_reports", L1, "metrics", "support"), {}),
    (("layer_reports", L3, "metrics", "scale_consistency", "coverage", "complete"), False),
    (("layer_reports", L3, "metrics", "scale_consistency", "coverage", "fraction"), 0.5),
    (("layer_reports", L3, "metrics", "functional_consistency", "coverage", "score_grounding", "defaulted_count"), 1),
    (("layer_reports", L3, "metrics", "functional_consistency", "coverage", "score_grounding"), None),
    (("layer_reports", L3, "metrics", "semantic_placement_consistency", "coverage", "score_grounding", "fraction"), 0.9),
    (("layer_reports", L3, "metrics", "style_consistency", "coverage", "score_grounding"), {}),
    (("layer_reports", L3, "metrics", "object_pairing_consistency", "coverage", "score_grounding", "defaulted_units"),
     [{"unit_id": "renderer_failed"}]),
])
def test_no_other_failure_can_be_waived_at_sufficient_coverage(accepted_pilot, path, value):
    report = deepcopy(accepted_pilot["report"])
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    decision = evaluate_report_acceptance(report, _policy())
    assert not decision["accepted"], path
    assert decision["reasons"]


def test_removing_metric_from_all_result_inventories_does_not_remove_obligation(accepted_pilot):
    report = deepcopy(accepted_pilot["report"])
    metric = "semantic_placement_consistency"
    for key in ("active_metrics_by_layer", "resolved_metrics_by_layer"):
        report["coverage"][key][L3].remove(metric)
    for key in ("active_metrics", "resolved_metrics"):
        report["layer_reports"][L3][key].remove(metric)
    report["scoring_reliability"]["metrics"] = [r for r in report["scoring_reliability"]["metrics"]
                                               if r["metric_id"] != f"{L3}.{metric}"]
    decision = evaluate_report_acceptance(report, _policy())
    assert "metric_inventory_differs_from_evaluation_plan" in decision["reasons"]


def test_active_l2_cannot_be_waived_as_frozen_asset_selection(accepted_pilot):
    report = deepcopy(accepted_pilot["report"])
    # A planned but unevaluated specification obligation must block even when
    # every L3 metric required by FrozenAssets has complete evidence.
    report["evaluation_plan"]["layers"][L2]["weight"] = 0.25
    report["evaluation_plan"]["layers"][L2]["metrics"]["oor"]["applicable"] = True
    assert not evaluate_report_acceptance(report, _policy())["accepted"]


@pytest.mark.parametrize("policy", [None, "unknown", 1, [], {}])
def test_unknown_policy_fails_closed(policy):
    with pytest.raises(ValueError, match="acceptance policy"):
        acceptance_policy_name({"acceptance_policy": policy})


@pytest.mark.parametrize("mode", ["native", "shared_db"])
def test_frozen_acceptance_rejects_other_protocols(mode):
    from test_generation_comparison_protocol import _catalog, _protocol
    protocol = _protocol(_catalog(), mode=mode).as_dict()
    protocol["evaluator"].update(_policy())
    with pytest.raises(ValueError, match="requires frozen_assets"):
        ComparisonProtocol.from_mapping(protocol)


def test_readiness_requires_explicit_opt_in_and_factual_ownership():
    strict = {"static_kwargs": {"asset_policy": dict(FROZEN_OWNERSHIP)}}
    assert not evaluator_policy_readiness(strict)["ready"]
    opted_in = evaluator_policy_readiness(_policy())
    assert opted_in["ready"] and not opted_in["scoring_modified"]
    assert set(opted_in["exempt_applicability_metrics"]) == EXEMPT_METRICS
    assert acceptance_policy_name({}) == COMPLETE_SCORE
    unowned = _policy()
    unowned["static_kwargs"].pop("asset_policy")
    assert not evaluator_policy_readiness(unowned)["ready"]


def test_acceptance_policy_is_hashed_but_not_generator_visible(accepted_pilot):
    fixture = accepted_pilot
    config = read_json(fixture["prepared"]["evaluator_config"])
    assert config["acceptance_policy"] == FROZEN_REQUIRED_METRICS
    assert "acceptance_policy" not in config["static_kwargs"]
    assert config["config_sha256"] == fixture["prepared"]["evaluator_config_sha256"]
    for case in fixture["prepared"]["cases"]:
        public_bytes = Path(case["generation_input"]).read_text()
        assert FROZEN_REQUIRED_METRICS not in public_bytes
        assert "evaluation_acceptance" not in public_bytes
        options = read_json(fixture["root"] / "pilot/cases" / case["case_id"] /
                            "catalog_placement/comparison/evaluation_input.json")
        assert "acceptance_policy" not in options


def test_acceptance_cannot_be_retrofitted_into_a_prepared_run(tmp_path, monkeypatch):
    from benchmark.generation_comparison.prepared import verify_prepared_artifacts
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
    )
    config = read_json(prepared["evaluator_config"])
    config["acceptance_policy"] = FROZEN_REQUIRED_METRICS
    write_json(prepared["evaluator_config"], config)
    monkeypatch.setattr("benchmark.generation_comparison.pilot.run_controlled_generation",
                        lambda **kwargs: pytest.fail("policy drift must not reach generation"))
    with pytest.raises(ValueError, match="hash"):
        verify_prepared_artifacts(tmp_path / "pilot", prepared)


def test_failed_iteration_prevents_final_only_acceptance(accepted_pilot):
    from benchmark.generation_comparison.pilot import _success_row
    fixture = accepted_pilot
    result = read_json(fixture["rows"][0]["run_manifest"])
    result["sceneweaver_trajectory"] = {"all_evaluations_accepted": False, "iterations": []}
    row = _success_row(
        case_manifest=fixture["prepared"]["cases"][0], method="scene_weaver",
        execution_mode="offline_native_artifact", result=result,
        evaluation=fixture["report"], evaluator_policy=_policy(),
    )
    assert row["evaluation_acceptance"]["accepted"]  # final report alone qualifies
    assert not row["evaluation_accepted"] and row["run_status"] == "failed"
    assert "trajectory_accepted=False" in row["failure_reason"]


def test_sceneweaver_each_state_uses_same_posthoc_acceptance(tmp_path, monkeypatch):
    from benchmark.generation_comparison.execution import run_controlled_generation
    from test_generation_comparison_protocol import (
        ALL_CONTROLS, _catalog, _generation_input, _native_output, _protocol,
    )
    state = _state()
    _install_external_fixtures(monkeypatch, state)
    catalog = _catalog()
    contract = _protocol(catalog, mode="frozen_assets").as_dict()
    contract["evaluator"].update(_policy())
    protocol = ComparisonProtocol.from_mapping(contract)

    def runner(*, method_input_path, out_dir, config):
        text = Path(method_input_path).read_text()
        assert FROZEN_REQUIRED_METRICS not in text
        assert "evaluation_acceptance" not in text
        assert "benchmark_score" not in text
        return _native_output("scene_weaver", out_dir, config)

    result = run_controlled_generation(
        generation_input=_generation_input(), adapter_name="scene_weaver",
        protocol=protocol, asset_catalog=catalog, out_dir=tmp_path / "trajectory",
        adapter_config={"runner": runner, "comparison_support": ALL_CONTROLS, "selected_iteration": 1},
        evaluation_kwargs=_policy()["static_kwargs"],
        evaluation_runtime=CanonicalEvaluationRuntime(_runtime_config(), require_credentials=False),
    )
    trajectory = result["sceneweaver_trajectory"]
    assert trajectory["valid_comparison_trajectory"]
    assert trajectory["all_evaluations_accepted"], trajectory
    assert not trajectory["benchmark_feedback_used_by_native_loop"]
    assert [row["iteration"] for row in trajectory["iterations"]] == [0, 1]
    assert len(state["renders"]) == 3  # final evaluation and two independent states
    for row in trajectory["iterations"]:
        assert row["benchmark_score_status"] == "partial_coverage"
        assert row["evaluation_acceptance"]["accepted_partial_coverage"]
        assert row["evaluation_acceptance"]["report_sha256"] == canonical_json_sha256(read_json(row["evaluation_report"]))
        assert row["selected_asset_ids"] == {"chair_0": "chair.asset.001"}
