from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.generation_comparison.pilot import (
    _trajectory_summary,
    prepare_controlled_pilot,
    run_prepared_pilot,
)
from benchmark.utils.io import read_json, write_json


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

    assert result["status"] == "completed"
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
