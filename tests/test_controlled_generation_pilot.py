from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmark.generation_comparison.pilot import (
    _execution_readiness,
    _upstream_execution_evidence,
    _trajectory_summary,
    bridge_execution_hashes,
    prepare_controlled_pilot,
    run_prepared_pilot,
)
from benchmark.utils.io import read_json, write_json


@pytest.mark.parametrize(
    ("record", "started"),
    [
        (None, False),
        ({"runner_kind": "subprocess", "return_code": None}, False),
        ({"runner_kind": "subprocess", "return_code": 0}, True),
        ({"runner_kind": "subprocess", "return_code": 3}, True),
        ({"runner_kind": "subprocess", "return_code": -9, "cancelled": True}, True),
        ({"runner_kind": "subprocess", "timed_out": True}, True),
        ({"runner_kind": "callback", "return_code": 0}, False),
        ({"runner_kind": "configured_native_artifact", "return_code": 0}, False),
    ],
)
def test_upstream_launch_claim_requires_process_evidence(tmp_path, record, started):
    if record is not None:
        write_json(tmp_path / "generator/execution/execution_result.json", record)
    assert _upstream_execution_evidence(tmp_path)["upstream_process_started"] is started


def test_bridge_execution_hashes_are_operator_reproducible(tmp_path: Path) -> None:
    entrypoint = tmp_path / "bridge.py"
    common = tmp_path / "_common.py"
    entrypoint.write_text("print('bridge')\n", encoding="utf-8")
    common.write_text("VALUE = 1\n", encoding="utf-8")

    first = bridge_execution_hashes(entrypoint)
    second = bridge_execution_hashes(entrypoint)

    assert first == second
    assert len(first["expected_entrypoint_sha256"]) == 64
    assert len(first["expected_bridge_bundle_sha256"]) == 64
    assert [item["name"] for item in first["bridge_bundle_files"]] == [
        "_common.py",
        "bridge.py",
    ]
    common.write_text("VALUE = 2\n", encoding="utf-8")
    changed = bridge_execution_hashes(entrypoint)
    assert changed["expected_entrypoint_sha256"] == first[
        "expected_entrypoint_sha256"
    ]
    assert changed["expected_bridge_bundle_sha256"] != first[
        "expected_bridge_bundle_sha256"
    ]


@pytest.mark.parametrize(
    ("method", "artifact_variable", "digest_variable", "expected_reason"),
    [
        (
            "layout_gpt",
            "layoutgpt_icl_examples",
            "layoutgpt_icl_sha256",
            "layoutgpt_icl_sha256_mismatch",
        ),
        (
            "scene_weaver",
            "frozen_plugin",
            "frozen_plugin_sha256",
            "sceneweaver_plugin_sha256_mismatch",
        ),
    ],
)
def test_readiness_hashes_actual_harness_support_file_bytes(
    method: str,
    artifact_variable: str,
    digest_variable: str,
    expected_reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / f"{method}_frozen.py"
    bridge.write_text("# bridge\n", encoding="utf-8")
    support_file = tmp_path / "support.json"
    support_file.write_text("{}\n", encoding="utf-8")
    source_pins = bridge_execution_hashes(bridge)
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "test-secret")
    report = _execution_readiness(
        method,
        {
            "adapter_config": {
                "execution": {
                    "repo_path": str(tmp_path / "missing-upstream"),
                    "expected_upstream_commit": "0" * 40,
                    "expected_entrypoint_sha256": source_pins[
                        "expected_entrypoint_sha256"
                    ],
                    "expected_bridge_bundle_sha256": source_pins[
                        "expected_bridge_bundle_sha256"
                    ],
                    "python_executable": sys.executable,
                    "command": [sys.executable, str(bridge)],
                    "template_variables": {
                        "bridge_script": str(bridge),
                        artifact_variable: str(support_file),
                        digest_variable: "0" * 64,
                    },
                    "environment": {
                        (
                            "LAYOUT_DDD_API_ENDPOINT"
                            if method == "layout_gpt"
                            else "LAYOUT_DDD_API_BASE_URL"
                        ): (
                            "http://127.0.0.1:9999/v1/chat/completions"
                            if method == "layout_gpt"
                            else "http://127.0.0.1:9999/v1"
                        )
                    },
                }
            }
        },
        offline_artifact=None,
        allow_offline_artifacts=False,
        required_layoutgpt_icl_sha256=("1" * 64 if method == "layout_gpt" else None),
    )
    assert expected_reason in report["reasons"]
    if method == "layout_gpt":
        assert "layoutgpt_icl_protocol_sha256_mismatch" in report["reasons"]


def test_readiness_requires_the_pinned_bridge_to_be_the_configured_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "layout_gpt_frozen.py"
    bridge.write_text("# pinned bridge\n", encoding="utf-8")
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("# unrelated runner\n", encoding="utf-8")
    icl = tmp_path / "icl.json"
    icl.write_text("[]\n", encoding="utf-8")
    source_pins = bridge_execution_hashes(bridge)
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "test-secret")
    report = _execution_readiness(
        "layout_gpt",
        {
            "adapter_config": {
                "execution": {
                    "repo_path": str(tmp_path / "missing-upstream"),
                    "expected_upstream_commit": "0" * 40,
                    **source_pins,
                    "python_executable": sys.executable,
                    "command": [sys.executable, str(unrelated)],
                    "template_variables": {
                        "bridge_script": str(bridge),
                        "layoutgpt_icl_examples": str(icl),
                        "layoutgpt_icl_sha256": "0" * 64,
                    },
                    "environment": {
                        "LAYOUT_DDD_API_ENDPOINT": (
                            "http://127.0.0.1:9999/v1/chat/completions"
                        )
                    },
                }
            }
        },
        offline_artifact=None,
        allow_offline_artifacts=False,
    )
    assert "bridge_script_not_in_command" in report["reasons"]


def test_layoutvlm_readiness_binds_preserved_scene_config_to_bridge_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = tmp_path / "layout_vlm_frozen.py"
    bridge.write_text("# bridge\n", encoding="utf-8")
    source_pins = bridge_execution_hashes(bridge)
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "test-secret")
    report = _execution_readiness(
        "layout_vlm",
        {
            "adapter_config": {
                "execution": {
                    "repo_path": str(tmp_path / "missing-upstream"),
                    "expected_upstream_commit": "0" * 40,
                    "expected_entrypoint_sha256": source_pins[
                        "expected_entrypoint_sha256"
                    ],
                    "expected_bridge_bundle_sha256": source_pins[
                        "expected_bridge_bundle_sha256"
                    ],
                    "python_executable": sys.executable,
                    "command": [
                        sys.executable,
                        str(bridge),
                        "--prepared-scene-config-output",
                        "{upstream_output_dir}/actual.json",
                    ],
                    "template_variables": {"bridge_script": str(bridge)},
                    "auxiliary_artifacts": {
                        "scene_config": "{upstream_output_dir}/different.json"
                    },
                    "environment": {
                        "LAYOUT_DDD_API_BASE_URL": "http://127.0.0.1:9999/v1"
                    },
                }
            }
        },
        offline_artifact=None,
        allow_offline_artifacts=False,
    )
    assert "layoutvlm_prepared_scene_config_path_mismatch" in report["reasons"]


def test_prepare_pilot_preflights_assets_cases_hashes_and_readiness(
    tmp_path: Path,
) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    result = prepare_controlled_pilot(
        spec=_pilot_spec(),
        asset_root=asset_root,
        out_dir=tmp_path / "pilot",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert result["status"] == "prepared"
    assert result["case_count"] == 5
    assert len(result["branch_commit"]) == 40
    preflight = read_json(result["asset_preflight"])
    assert preflight["status"] == "passed"
    assert preflight["passed"] == 1
    assert preflight["assets"][0]["canonical_front_status"] == (
        "unavailable_not_invented"
    )
    catalog = read_json(result["catalog"])
    assert len(catalog["catalog_sha256"]) == 64
    assert catalog["assets"][0]["content"]["mesh_sha256"] == preflight[
        "assets"
    ][0]["mesh_sha256"]
    assert [row["complexity"]["object_count"] for row in result["cases"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert result["cases"][1]["complexity"] == {
        "object_count": 2,
        "room_area": 20.0,
        "object_density": 0.1,
        "pairwise_interaction_proxy": 1,
    }
    compatibility = read_json(result["compatibility_report"])
    statuses = {row["method"]: row["status"] for row in compatibility["methods"]}
    assert statuses == {
        "catalog_placement": "SEMANTICALLY_ELIGIBLE_INFRASTRUCTURE_UNAVAILABLE",
        "layout_gpt": "INELIGIBLE",
        "layout_vlm": "SEMANTICALLY_ELIGIBLE_INFRASTRUCTURE_UNAVAILABLE",
    }
    evaluator = read_json(result["evaluator_config"])
    assert evaluator["config_sha256"] == result["evaluator_config_sha256"]
    assert len({row["protocol_sha256"] for row in result["cases"]}) == 5


def test_asset_category_mismatch_fails_before_generation(tmp_path: Path) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    spec = _pilot_spec()
    spec["catalog"]["assets"][0]["category"] = "table"
    for case in spec["cases"]:
        for slot in case["objects"]:
            slot["category"] = "table"
    with pytest.raises(ValueError, match="asset preflight failed"):
        prepare_controlled_pilot(
            spec=spec,
            asset_root=asset_root,
            out_dir=tmp_path / "invalid",
        )
    report = read_json(tmp_path / "invalid" / "asset_preflight.json")
    assert report["status"] == "failed"
    assert "category_mismatch" in report["assets"][0]["errors"]
    assert not (tmp_path / "invalid" / "pilot_manifest.json").exists()


def test_preflight_enforces_frozen_catalog_and_source_asset_hashes(
    tmp_path: Path,
) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    csv_path = asset_root / "imaginarium_asset_info.csv"
    fbx_path = asset_root / "chair_asset" / "chair_asset.fbx"
    metadata_path = asset_root / "chair_asset" / "chair_asset_metadata.json"
    spec = _pilot_spec()
    spec["catalog"]["source_catalog_csv_sha256"] = hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    spec["catalog"]["assets"][0].update(
        {
            "source_fbx_sha256": hashlib.sha256(fbx_path.read_bytes()).hexdigest(),
            "source_metadata_sha256": hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest(),
        }
    )
    prepare_controlled_pilot(
        spec=spec,
        asset_root=asset_root,
        out_dir=tmp_path / "valid",
    )

    csv_mismatch = _pilot_spec()
    csv_mismatch["catalog"]["source_catalog_csv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalog CSV differs"):
        prepare_controlled_pilot(
            spec=csv_mismatch,
            asset_root=asset_root,
            out_dir=tmp_path / "csv_mismatch",
        )

    asset_mismatch = _pilot_spec()
    asset_mismatch["catalog"]["assets"][0]["source_fbx_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="asset preflight failed"):
        prepare_controlled_pilot(
            spec=asset_mismatch,
            asset_root=asset_root,
            out_dir=tmp_path / "asset_mismatch",
        )
    report = read_json(tmp_path / "asset_mismatch" / "asset_preflight.json")
    assert "source_fbx_hash_mismatch" in report["assets"][0]["errors"]


def test_offline_dry_run_generates_tables_without_claiming_real_execution(
    tmp_path: Path,
) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=asset_root,
        out_dir=tmp_path / "pilot",
        repo_root=Path(__file__).resolve().parents[1],
    )
    native = write_json(
        tmp_path / "native_catalog_placement.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair_instance",
                    "asset_id": "chair_asset",
                    "center_m": [2.0, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_0",
                }
            ],
        },
    )
    result = run_prepared_pilot(
        prepared_dir=tmp_path / "pilot",
        method_outputs={
            "catalog_placement": {
                prepared["cases"][0]["case_id"]: native,
            }
        },
        dry_run_only=True,
        allow_offline_artifacts=True,
    )

    # Conversion success without complete evaluator coverage is not a completed
    # experiment (Pro F6); retain the artifact and the unscored result.
    assert result["status"] == "failed"
    assert result["experiment_complete"] is False
    assert result["attempted_runs"] == 1
    assert result["valid_runs"] == 1
    assert result["real_upstream_execution_performed"] is False
    rows = [
        read_json_line
        for read_json_line in (
            json.loads(line)
            for line in (tmp_path / "pilot" / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    assert rows[0]["execution_mode"] == "offline_artifact"
    assert rows[0]["valid_comparison_run"] is True
    assert rows[0]["evaluation_success"] is False
    assert rows[0]["score_available"] is False
    assert rows[0]["failure_class"] == "evaluator_infrastructure_failure"
    assert rows[0]["evaluator_config_hash"] == result["evaluator_config_sha256"]
    with (tmp_path / "pilot" / "results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["case_id"] == prepared["cases"][0]["case_id"]
    summary = read_json(result["summary"])
    assert summary["methods"]["catalog_placement"]["attempted_cases"] == 1
    assert summary["methods"]["catalog_placement"]["valid_runs"] == 1
    assert summary["methods"]["catalog_placement"]["scored_runs"] == 0
    assert summary["statistical_significance_tested"] is False


def test_pending_human_asset_selection_can_prepare_but_cannot_run(
    tmp_path: Path,
) -> None:
    spec = _pilot_spec(methods=["catalog_placement"])
    spec["asset_selection_status"] = "candidate_pending_human_approval"
    prepared = prepare_controlled_pilot(
        spec=spec,
        asset_root=_asset_root(tmp_path / "assets"),
        out_dir=tmp_path / "pilot",
    )

    assert prepared["asset_selection_status"] == (
        "candidate_pending_human_approval"
    )
    with pytest.raises(
        ValueError, match="frozen asset selection is not approved for generation"
    ):
        run_prepared_pilot(
            prepared_dir=tmp_path / "pilot",
            method_configs={},
            dry_run_only=True,
        )
    assert not (tmp_path / "pilot" / "results.jsonl").exists()


def test_case_audit_provenance_is_not_generator_visible(tmp_path: Path) -> None:
    spec = _pilot_spec(methods=["catalog_placement"])
    spec["cases"][0]["source_provenance"] = {
        "source_model_label": "baseline-model",
        "displayed_liveboard_score": 99.9,
        "source_blend": "/private/audit/path.blend",
        "reference_annotation": {"hidden": True},
    }
    prepared = prepare_controlled_pilot(
        spec=spec,
        asset_root=_asset_root(tmp_path / "assets"),
        out_dir=tmp_path / "pilot",
    )
    generation_input = read_json(prepared["cases"][0]["generation_input"])
    assert "source_provenance" not in generation_input["scene_request"]["metadata"]
    manifest = read_json(prepared["cases"][0]["case_manifest"])
    assert manifest["source_provenance"] == spec["cases"][0]["source_provenance"]


def test_prepare_and_run_refuse_to_overwrite_existing_artifacts(
    tmp_path: Path,
) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=asset_root,
        out_dir=tmp_path / "pilot",
    )
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        prepare_controlled_pilot(
            spec=_pilot_spec(methods=["catalog_placement"]),
            asset_root=asset_root,
            out_dir=tmp_path / "pilot",
        )
    native = write_json(
        tmp_path / "native.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair_instance",
                    "asset_id": "chair_asset",
                    "center_m": [2.0, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_0",
                }
            ],
        },
    )
    kwargs = {
        "prepared_dir": tmp_path / "pilot",
        "method_outputs": {
            "catalog_placement": {prepared["cases"][0]["case_id"]: native}
        },
        "dry_run_only": True,
        "allow_offline_artifacts": True,
    }
    run_prepared_pilot(**kwargs)
    with pytest.raises(FileExistsError, match="already exist"):
        run_prepared_pilot(**kwargs)


def test_pilot_persists_asset_identity_failure_as_method_failure(
    tmp_path: Path,
) -> None:
    asset_root = _asset_root(tmp_path / "assets")
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=asset_root,
        out_dir=tmp_path / "pilot",
    )
    native = write_json(
        tmp_path / "wrong_asset.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair_instance",
                    "asset_id": "replacement_asset",
                    "center_m": [2.0, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_0",
                }
            ],
        },
    )
    result = run_prepared_pilot(
        prepared_dir=tmp_path / "pilot",
        method_outputs={
            "catalog_placement": {prepared["cases"][0]["case_id"]: native}
        },
        dry_run_only=True,
        allow_offline_artifacts=True,
    )
    row = json.loads(
        (tmp_path / "pilot" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert result["valid_runs"] == 0
    assert row["failure_class"] == "asset_identity_violation"
    assert row["failure_source"] == "method"
    failure = read_json(
        tmp_path
        / "pilot"
        / "cases"
        / prepared["cases"][0]["case_id"]
        / "catalog_placement"
        / "pilot_failure.json"
    )
    assert failure["failure_class"] == "asset_identity_violation"


def test_sceneweaver_trajectory_summary_uses_only_evaluator_reports(
    tmp_path: Path,
) -> None:
    rows = []
    for iteration, (score, hard_scores) in enumerate(
        [
            (0.4, [0.0, 1.0, 1.0]),
            (0.3, [0.0, 0.0, 1.0]),
            (0.7, [1.0, 1.0, 1.0]),
        ]
    ):
        report = write_json(
            tmp_path / f"report_{iteration}.json",
            {
                "benchmark_score": score,
                "reports": {
                    "generic_validity": {
                        "metrics": {
                            name: {"status": "checked", "score": metric_score}
                            for name, metric_score in zip(
                                ("collision", "support", "oob"), hard_scores
                            )
                        }
                    }
                },
            },
        )
        rows.append(
            {
                "iteration": iteration,
                "benchmark_score": score,
                "evaluation_report": report.as_posix(),
            }
        )
    summary = _trajectory_summary({"iterations": rows})
    assert summary == {
        "initial_score": 0.4,
        "final_score": 0.7,
        "score_delta": pytest.approx(0.3),
        "iteration_count": 3,
        "success_at_iteration": 2,
        "trajectory_regression_count": 1,
        "trajectory_hard_failure_fixes": 2,
        "trajectory_hard_failure_regressions": 1,
    }


@pytest.mark.parametrize("artifact", [
    "generation_input", "evaluation_object_plan", "protocol", "evaluator_config",
    "catalog", "case_manifest",
])
def test_prepared_artifact_drift_rejects_before_any_generation(
    artifact: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
    )
    # Corrupt a later case as well: the gate must check the entire planned
    # cohort before spending a call on the first case.
    row = prepared if artifact in {"evaluator_config", "catalog"} else prepared["cases"][-1]
    path = Path(row[artifact])
    content = read_json(path)
    content["unexpected_drift"] = True
    write_json(path, content)
    calls = []
    monkeypatch.setattr(
        "benchmark.generation_comparison.pilot.run_controlled_generation",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(ValueError, match="prepared artifact hash mismatch"):
        run_prepared_pilot(prepared_dir=tmp_path / "pilot")
    assert calls == []
    assert read_json(prepared["manifest_path"])["status"] == "prepared"


def test_blocked_units_are_all_reported_and_cli_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    from benchmark.generation_comparison.pilot import main

    prepare_controlled_pilot(
        spec=_pilot_spec(), asset_root=_asset_root(tmp_path / "assets"),
        out_dir=tmp_path / "pilot",
    )
    config = write_json(tmp_path / "methods.json", {})
    monkeypatch.setattr(sys, "argv", [
        "pilot", "run", "--prepared-dir", str(tmp_path / "pilot"),
        "--method-configs", str(config),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert result["attempted_runs"] == 0
    assert result["planned_runs"] == 15
    assert not result["real_upstream_execution_performed"]
    rows = [json.loads(line) for line in (tmp_path / "pilot/results.jsonl").read_text().splitlines()]
    assert len(rows) == 15
    assert all(row["run_status"] == "blocked" and row["readiness"] for row in rows)


def test_cancellation_is_propagated_and_does_not_start_next_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
    )
    native = write_json(tmp_path / "native.json", {})
    calls = []

    def cancel(**kwargs):
        calls.append(kwargs)
        raise KeyboardInterrupt("operator stop")

    monkeypatch.setattr("benchmark.generation_comparison.pilot.run_controlled_generation", cancel)
    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        run_prepared_pilot(
            prepared_dir=tmp_path / "pilot", allow_offline_artifacts=True,
            method_outputs={"catalog_placement": {row["case_id"]: native for row in prepared["cases"]}},
        )
    assert len(calls) == 1
    result = read_json(prepared["manifest_path"])
    assert result["status"] == "cancelled"
    assert result["attempted_runs"] == 1
    assert result["unattempted_runs"] == 4
    rows = [json.loads(line) for line in (tmp_path / "pilot/results.jsonl").read_text().splitlines()]
    assert [row["run_status"] for row in rows] == ["cancelled"] + ["skipped"] * 4


def test_prelaunch_output_conflict_preserves_old_files_and_finishes_plan(tmp_path):
    prepared = prepare_controlled_pilot(
        spec=_pilot_spec(methods=["catalog_placement"]),
        asset_root=_asset_root(tmp_path / "assets"), out_dir=tmp_path / "pilot",
    )
    native = write_json(tmp_path / "native.json", {})
    old = write_json(tmp_path / "pilot/cases/case_001/catalog_placement/comparison/run_manifest.json",
                     {"old_run": "must not be attributed or overwritten"})
    before = old.read_bytes()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_prepared_pilot(
            prepared_dir=tmp_path / "pilot", allow_offline_artifacts=True,
            method_outputs={"catalog_placement": {row["case_id"]: native for row in prepared["cases"]}},
        )
    assert old.read_bytes() == before
    result = read_json(prepared["manifest_path"])
    assert result["status"] == "blocked" and result["planned_runs"] == 5
    rows = [json.loads(line) for line in (tmp_path / "pilot/results.jsonl").read_text().splitlines()]
    assert [row["run_status"] for row in rows] == ["blocked"] + ["skipped"] * 4
    assert all(not row["attempted"] and row["run_manifest"] is None for row in rows)


def _asset_root(root: Path) -> Path:
    name = "chair_asset"
    directory = root / name
    directory.mkdir(parents=True)
    (directory / f"{name}.fbx").write_bytes(b"synthetic fbx fixture")
    write_json(
        directory / f"{name}_metadata.json",
        {
            "transformed_size": [0.8, 0.7, 1.0],
            "transformed_bbox_center": [0.0, 0.0, 0.0],
            "is_coordinate_transformed": True,
        },
    )
    with (root / "imaginarium_asset_info.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "name_en",
                "category",
                "short_desc",
                "scaling_strategy",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "name_en": name,
                "category": "chair",
                "short_desc": "fixture chair",
                "scaling_strategy": "ISOTROPIC",
            }
        )
    return root


def _pilot_spec(methods: list[str] | None = None) -> dict:
    cases = []
    for index in range(5):
        count = index + 1
        cases.append(
            {
                "case_id": f"case_{count:03d}",
                "scene_type": "office",
                "seed": 100 + index,
                "room": {
                    "boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
                    "height": 3.0,
                    "unit": "meter",
                },
                "instruction": "Arrange the frozen fixture chairs at native scale 1.0.",
                "objects": [
                    {
                        "slot_id": f"chair_{slot}",
                        "category": "chair",
                        "description": "fixture chair",
                        "asset_id": "chair_asset",
                    }
                    for slot in range(count)
                ],
            }
        )
    return {
        "schema_version": "controlled_generation_pilot_v1",
        "pilot_id": "fixture_pilot",
        "label": "pilot / integration validation",
        "protocol_id": "generation_comparison_v1",
        "protocol_version": 1,
        "mode": "frozen_assets",
        "catalog": {
            "catalog_id": "fixture_catalog",
            "catalog_version": "1",
            "source_db": "fixture",
            "assets": [
                {
                    "asset_id": "chair_asset",
                    "category": "chair",
                    "description": "fixture chair",
                }
            ],
        },
        "evaluator": {
            "policy": "same_canonical_run_evaluate",
            "profile": "repository_default_canonical_l0_l4",
            "evidence_policy": "repository_default_no_injected_evidence",
            "rendering_policy": "repository_default_no_forced_render",
            "static_kwargs": {},
        },
        "generation": {
            "budget_policy": "method_native_recorded",
            "scale_policy": "fixed_native_scale",
        },
        "methods": methods or ["catalog_placement", "layout_gpt", "layout_vlm"],
        "cases": cases,
    }
