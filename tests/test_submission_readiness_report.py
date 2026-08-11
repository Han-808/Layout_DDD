from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import benchmark.api.submission as submission_module
import benchmark.materialization.preparation as preparation_module
import benchmark.materialization.readiness as readiness_module
from benchmark.api.submission import (
    TrustedCaseBundle,
    evaluate_prepared_submission,
)
from benchmark.evaluator.profile import resolve_evaluation_profile
from benchmark.io_contracts import O3_SCENE_PACKAGE
from benchmark.materialization import MaterializationResult
from benchmark.materialization.catalog import FrozenCatalog, sha256_file
from benchmark.materialization.catalog import sha256_json
from benchmark.nl_scene.generation_input import (
    build_generation_input,
    build_scene_request,
)
from benchmark.materialization.readiness import (
    L0,
    L1,
    L2,
    L3,
    L4,
    build_not_evaluable_evaluation_report,
    build_readiness_report,
)
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.utils.io import read_json
from benchmark.utils.io import write_json


ROOT = Path(__file__).resolve().parents[1]


def _failed_readiness() -> dict:
    return build_readiness_report(
        status="not_evaluable",
        reason_codes=["registry_scene_asset_id_mismatch"],
        failure_stage="materialization_consistency",
        failure_owner="generator",
        checks=[
            {
                "id": "artifact_schema",
                "passed": True,
                "detail": "catalog_placement_v1",
            },
            {
                "id": "materialization_consistency",
                "passed": False,
                "reason_codes": ["registry_scene_asset_id_mismatch"],
                "detail": {"instance_id": "chair_left"},
            },
        ],
        provenance={
            "source_artifact_sha256": "a" * 64,
            "adapter_contract_revision": "catalog_placement_v1",
        },
    )


def _trusted_bundle(tmp_path: Path) -> TrustedCaseBundle:
    manifest_path = tmp_path / "case_bundle.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return TrustedCaseBundle(
        root=tmp_path,
        manifest_path=manifest_path,
        manifest_sha256="c" * 64,
        case_id="blocked_case",
        evaluator_output_type=O3_SCENE_PACKAGE,
        scene_request={
            "request_id": "blocked_request",
            "instruction": "Place one frozen object.",
            "scene_type": "room",
            "structure": True,
            "prompt_granularity": "fine_grained",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
                "unit": "meter",
            },
        },
        reference_annotation=None,
        specification_contract=None,
        specification_activation_mode="none",
        functional_semantic_config=None,
        scene_quality_config=None,
        object_grouping_report=None,
        asset_policy=None,
        authorized_deviations=None,
        spatial_fidelity_ontology=None,
        visual_style_spec=None,
        evaluation_profile=resolve_evaluation_profile(),
        workflow="canonical_l0_l4",
        enabled_evaluators={},
        p0b_official_mode=True,
        camera_evidence={
            "mode": None,
            "metric_modes": {},
            "max_views": 2,
            "max_steps": 0,
            "collision_overlay": True,
            "collision_contour": True,
            "active_fallback": {"enabled": False},
        },
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        artifact_records={},
    )


def _trusted_generation_input() -> dict:
    scene_request = build_scene_request(
        request_id="blocked_request",
        instruction="Place one frozen object.",
        scene_type="room",
        room={
            "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "height": 3.0,
            "unit": "meter",
        },
        structure=True,
    )
    return build_generation_input(
        scene_request=scene_request,
        object_plan={
            "request_id": "blocked_request",
            "scene_type": "room",
            "scene_description": "one chair",
            "objects": [
                {
                    "id": "chair_slot",
                    "role": "chair",
                    "category": "chair",
                    "description": "one chair",
                    "count": 1,
                    "placement_intent": {
                        "absolute_relations": [],
                        "relative_relations": [],
                    },
                    "metadata": {},
                }
            ],
            "global_constraints": [],
            "relations": [],
        },
        asset_selection={
            "request_id": "blocked_request",
            "objects": [
                {
                    "object_id": "chair_slot",
                    "object_spec": {
                        "role": "chair",
                        "category": "chair",
                        "description": "one chair",
                        "estimated_size": [1.0, 1.0, 1.0],
                        "count": 1,
                    },
                    "retrieval_query": {
                        "description": "one chair",
                        "category": "chair",
                        "size_constraint": [1.0, 1.0, 1.0],
                    },
                    "selected_asset": {
                        "jid": "asset",
                        "category": "chair",
                        "retrieval_category": "chair",
                        "desc": "a frozen chair",
                        "short_desc": "chair",
                        "size": [1.0, 1.0, 1.0],
                        "asset_ref": {
                            "source_db": "test",
                            "asset_key": "asset",
                            "mesh_uri": None,
                            "pointcloud_uri": None,
                            "metadata_uri": None,
                        },
                        "asset_proxy": {
                            "type": "canonical_catalog_bbox",
                            "bbox_center_local": [0.0, 0.0, 0.0],
                            "bbox_size": [1.0, 1.0, 1.0],
                        },
                        "metadata": {},
                    },
                    "candidates": [],
                    "selection_action": "select",
                    "selection_decision": {
                        "action": "select",
                        "selected_jid": "asset",
                        "reason": "fixture",
                        "generation_request": None,
                    },
                    "selection_reason": "fixture",
                }
            ],
        },
        evaluator_output_type=O3_SCENE_PACKAGE,
    )


def _catalog_bound_stub(
    tmp_path: Path,
) -> tuple[MaterializationResult, Path, Path]:
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "prepared_assets.csv"
    asset_csv.write_text("name_en\nasset\n", encoding="utf-8")
    catalog_hash = sha256_file(asset_csv)
    provenance_core_path = write_json(
        prepared_root / "provenance_core.json",
        {
            "provenance_core_version": (
                "catalog_materialization_provenance_core_v1"
            ),
            "adapter_contract_revision": "catalog_placement_v1",
            "materialization_revision": "fixed_catalog_materialization_v1",
            "case_id": "blocked_case",
            "case_bundle_manifest_sha256": "c" * 64,
            "catalog_snapshot_id": "catalog_v1",
            "catalog_csv_path": asset_csv.resolve().as_posix(),
            "asset_root_path": asset_root.resolve().as_posix(),
            "representation_hashes": {
                "catalog_csv_sha256": catalog_hash,
            },
        },
    )
    provenance_core_hash = sha256_file(provenance_core_path)
    provenance_path = write_json(
        prepared_root / "provenance.json",
        {
            "catalog": {
                "snapshot_id": "catalog_v1",
                "asset_csv_path": asset_csv.resolve().as_posix(),
                "asset_root_path": asset_root.resolve().as_posix(),
                "catalog_csv_sha256": catalog_hash,
            },
            "artifacts": {
                "provenance_core": provenance_core_path.as_posix(),
            },
        },
    )
    return (
        MaterializationResult(
            normalized_scene_path=prepared_root / "normalized_scene.json",
            instance_registry_path=prepared_root / "instance_registry.json",
            trusted_render_source_path=prepared_root / "evaluation.blend",
            consistency_report_path=prepared_root / "consistency_report.json",
            readiness_report_path=prepared_root / "readiness_report.json",
            provenance_path=provenance_path,
            hashes={
                "catalog_csv_sha256": catalog_hash,
                "provenance_core_sha256": provenance_core_hash,
            },
        ),
        asset_root,
        asset_csv,
    )


def test_build_readiness_report_has_only_the_gate_contract_fields() -> None:
    report = build_readiness_report(
        status="ready",
        reason_codes=[],
        failure_owner=None,
        checks={
            "artifact_schema": True,
            "catalog_resolution": {
                "passed": True,
                "detail": {"snapshot_id": "catalog_v1"},
            },
        },
        provenance={"case_id": "case"},
    )

    assert set(report) == {
        "gate_version",
        "status",
        "reason_codes",
        "failure_stage",
        "primary_failure_owner",
        "contributing_owners",
        "failure_owner",
        "checks",
        "provenance",
    }
    assert report["gate_version"] == "submission_readiness_v1"
    assert report["status"] == "ready"
    assert report["reason_codes"] == []
    assert report["failure_stage"] is None
    assert report["primary_failure_owner"] is None
    assert report["contributing_owners"] == []
    assert report["failure_owner"] is None
    assert [check["id"] for check in report["checks"]] == [
        "artifact_schema",
        "catalog_resolution",
    ]
    assert all(check["passed"] for check in report["checks"])


def test_readiness_is_fail_closed_for_empty_or_failed_checks() -> None:
    empty = build_readiness_report(
        status="ready",
        checks=[],
        provenance={},
    )
    failed = build_readiness_report(
        status="ready",
        checks=[
            {
                "id": "trusted_blend_hash",
                "passed": False,
                "failure_owner": "infrastructure",
            }
        ],
        provenance={},
    )

    assert empty["status"] == "not_evaluable"
    assert empty["reason_codes"] == ["readiness_checks_missing"]
    assert failed["status"] == "not_evaluable"
    assert failed["reason_codes"] == ["trusted_blend_hash_failed"]
    assert failed["failure_stage"] == "submission_readiness"
    assert failed["primary_failure_owner"] == "infrastructure"
    assert failed["contributing_owners"] == []
    assert failed["failure_owner"] == "infrastructure"


def test_readiness_attributes_primary_and_contributing_owners() -> None:
    report = build_readiness_report(
        status="not_evaluable",
        failure_stage="evaluation_time_trust_audit",
        primary_failure_owner="benchmark",
        contributing_owners=["submission", "benchmark"],
        checks=[
            {
                "id": "trusted_catalog",
                "passed": False,
                "failure_owner": "submission",
                "reason_codes": ["catalog_binding_mismatch"],
            },
            {
                "id": "trusted_renderer",
                "passed": False,
                "failure_owner": "benchmark",
                "reason_codes": ["renderer_unavailable"],
            },
        ],
    )

    assert report["failure_stage"] == "evaluation_time_trust_audit"
    assert report["primary_failure_owner"] == "benchmark"
    assert report["contributing_owners"] == ["submission"]
    assert report["failure_owner"] == "benchmark"
    assert report["reason_codes"] == [
        "catalog_binding_mismatch",
        "renderer_unavailable",
    ]


def test_legacy_failure_owner_is_normalized_into_structured_attribution() -> None:
    normalized = readiness_module._normalize_readiness(
        {
            "gate_version": "submission_readiness_v1",
            "status": "not_evaluable",
            "reason_codes": ["legacy_failure"],
            "failure_owner": "generator",
            "checks": [
                {
                    "id": "legacy_check",
                    "passed": False,
                    "reason_codes": ["legacy_failure"],
                }
            ],
            "provenance": {},
        }
    )

    assert normalized["failure_stage"] == "submission_readiness"
    assert normalized["primary_failure_owner"] == "generator"
    assert normalized["contributing_owners"] == []
    assert normalized["failure_owner"] == "generator"


def test_readiness_rejects_unknown_status_and_duplicate_checks() -> None:
    with pytest.raises(ValueError, match="ready.*not_evaluable"):
        build_readiness_report(status="failed", checks={"schema": False})
    with pytest.raises(ValueError, match="duplicated"):
        build_readiness_report(
            checks=[
                {"id": "schema", "passed": True},
                {"id": "schema", "passed": True},
            ]
        )


def test_not_evaluable_report_is_canonical_and_schema_valid() -> None:
    bundle = {
        "case_id": "case",
        "manifest_sha256": "b" * 64,
        "evaluator_output_type": "o3_scene_package",
        "catalog_snapshot_id": "catalog_v1",
        "artifact_records": {},
        "scene_request": {
            "request_id": "request",
            "prompt_granularity": "fine_grained",
        },
        "specification_contract": {
            "claims": {
                "oor": [],
                "oar": [],
                "functional_semantic_fidelity": [],
            }
        },
    }
    readiness = _failed_readiness()
    report = build_not_evaluable_evaluation_report(
        readiness=readiness,
        bundle=bundle,
        scene_id="scene",
    )

    assert report["evaluation_status"] == "not_evaluable"
    assert report["benchmark_score"] is None
    assert report["benchmark_score_100"] is None
    assert report["benchmark_score_status"] == "not_evaluable"
    assert report["scoring_profile"]["scoring_profile_id"] == (
        "intrinsic_validity_v1"
    )
    assert report["canonical_object_denominator"] == {
        "ordered_object_ids": [],
        "n_scene": 0,
    }
    assert report["scoring_reliability"]["schema_version"] == (
        "scoring_reliability_v2"
    )
    assert report["official_submission"] is False
    assert report["protocol_scope"] == "official_submission"
    assert tuple(report["layer_reports"]) == (L0, L1, L2, L3, L4)
    assert report["category_reports"] == report["layer_reports"]
    assert report["layer_reports"][L0] == {
        "layer": L0,
        "status": "not_evaluable",
        "score": None,
        "affects_score": False,
        "checks": ["artifact_schema", "materialization_consistency"],
        "reason": "registry_scene_asset_id_mismatch",
        "readiness": readiness,
    }
    for layer in (L1, L2, L3):
        assert report["layer_reports"][layer]["status"] == "incomplete"
        assert report["layer_reports"][layer]["score"] is None
        assert report["layer_reports"][layer]["metrics"] == {}
        assert report["layer_reports"][layer]["resolved_metrics"] == []
    assert report["reports"]["generic_validity"]["status"] == "not_run"
    assert report["reports"]["scene_quality"]["status"] == "not_run"
    assert report["coverage"]["covered_layers"] == []
    assert report["coverage"]["complete"] is False
    assert "|scoring_profile:intrinsic_validity_v1|" in report[
        "coverage"
    ]["comparability_signature"]
    assert "|scoring_spec:object_equivalent_burden_v1" in report[
        "coverage"
    ]["comparability_signature"]
    assert report["evidence_provenance"]["render_evidence"] == "not_generated"

    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_not_evaluable_builder_supports_wrapper_keyword_signature() -> None:
    report = build_not_evaluable_evaluation_report(
        readiness=_failed_readiness(),
        scene_id=None,
        request_id="request",
        prompt_granularity="coarse_grained",
        evaluation_profile=None,
    )

    assert report["protocol_scope"] == "diagnostic_evaluation_api"
    assert report["request_id"] == "request"
    assert report["prompt_granularity"] == "coarse_grained"
    assert "case_bundle" not in report
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)


def test_existing_success_l0_without_readiness_remains_schema_compatible() -> None:
    report = build_not_evaluable_evaluation_report(
        readiness=_failed_readiness(),
        request_id="request",
    )
    compatible_success = deepcopy(report)
    compatible_success["evaluation_status"] = "complete"
    compatible_success["benchmark_score"] = 0.5
    compatible_success["benchmark_score_status"] = "complete"
    compatible_success["layer_reports"][L0] = {
        "layer": L0,
        "status": "passed",
        "score": None,
        "affects_score": False,
        "checks": ["schema"],
        "reason": None,
    }
    compatible_success["category_reports"] = deepcopy(
        compatible_success["layer_reports"]
    )

    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(compatible_success)


def test_legacy_readiness_v1_without_additive_attribution_fields_validates() -> None:
    report = build_not_evaluable_evaluation_report(
        readiness=_failed_readiness(),
        request_id="request",
    )

    def strip_additive_fields(value) -> None:
        if isinstance(value, dict):
            if value.get("gate_version") == "submission_readiness_v1":
                value.pop("failure_stage", None)
                value.pop("primary_failure_owner", None)
                value.pop("contributing_owners", None)
            for child in value.values():
                strip_additive_fields(child)
        elif isinstance(value, list):
            for child in value:
                strip_additive_fields(child)

    strip_additive_fields(report)
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    ("message", "source_kind", "expected_stage"),
    [
        (
            "generator artifact JSON must be an object",
            "json_file",
            "source_parsing",
        ),
        (
            "instances[0].uniform_scale must be finite",
            "json_file",
            "generator_contract_validation",
        ),
        (
            "instances[0].slot_id is not a public slot",
            "json_file",
            "slot_binding",
        ),
        (
            "unknown frozen catalog asset_id chair_x",
            "json_file",
            "asset_resolution",
        ),
        (
            "read-only blend inspector exited with status 1",
            "native_blend",
            "native_inspection",
        ),
        (
            "public native instance mapping is not valid JSON",
            "native_blend",
            "source_parsing",
        ),
        (
            "public native instance mapping has invalid root fields",
            "native_blend",
            "generator_contract_validation",
        ),
        (
            "catalog materializer exited with status 1",
            "json_file",
            "materialization",
        ),
    ],
)
def test_preparation_failure_stage_is_narrowly_attributed(
    message: str,
    source_kind: str,
    expected_stage: str,
) -> None:
    readiness = preparation_module._failure_readiness(
        preparation_module.MaterializationError(message),
        provenance={
            "adapter_contract_revision": "catalog_placement_v1",
            "case_bundle_manifest_sha256": "a" * 64,
            "source": {"kind": source_kind},
        },
    )

    assert readiness["failure_stage"] == expected_stage


def test_readiness_module_has_no_evaluator_import_boundary() -> None:
    tree = ast.parse(Path(readiness_module.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not {
        module
        for module in imported_modules
        if module == "benchmark.evaluator"
        or module.startswith("benchmark.evaluator.")
        or module == "benchmark.api.evaluation"
    }


def test_readiness_failure_calls_no_renderer_gate_judge_or_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "renderer": 0,
        "evidence_gate": 0,
        "judge": 0,
        "metrics": 0,
    }

    class _ForbiddenRenderer:
        def render_prepared_scene(self, **_: Any) -> dict:
            calls["renderer"] += 1
            raise AssertionError("renderer must not run before readiness")

    class _ForbiddenJudge:
        def __getattr__(self, _: str):
            def forbidden(*args: Any, **kwargs: Any) -> dict:
                del args, kwargs
                calls["judge"] += 1
                raise AssertionError("judge must not run before readiness")

            return forbidden

    def forbidden_metrics(*args: Any, **kwargs: Any) -> dict:
        del args, kwargs
        calls["metrics"] += 1
        raise AssertionError("metrics must not run before readiness")

    def forbidden_evidence_gate(*args: Any, **kwargs: Any):
        del args, kwargs
        calls["evidence_gate"] += 1
        raise AssertionError("EvidenceGate must not run before readiness")

    monkeypatch.setattr(submission_module, "run_evaluate", forbidden_metrics)
    monkeypatch.setattr(
        DeterministicEvidenceGate,
        "check",
        forbidden_evidence_gate,
    )

    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    failed_readiness = _failed_readiness()
    readiness_path = write_json(
        prepared_root / "readiness_report.json",
        failed_readiness,
    )
    consistency_path = write_json(
        prepared_root / "consistency_report.json",
        {
            "gate_version": "materialization_consistency_v1",
            "status": "failed",
            "checks": {},
            "mismatches": [{"code": "registry_scene_asset_id_mismatch"}],
            "hashes": {},
        },
    )
    provenance_path = write_json(
        prepared_root / "provenance.json",
        {
            "provenance_version": "catalog_materialization_provenance_v1",
            "adapter_contract_revision": "catalog_placement_v1",
            "materialization_revision": "fixed_catalog_materialization_v1",
            "case_id": "blocked_case",
            "case_bundle_manifest_sha256": "c" * 64,
            "catalog_snapshot_id": "catalog_v1",
            "status": "not_evaluable",
            "artifacts": {},
            "hashes": {},
        },
    )
    prepared = MaterializationResult(
        normalized_scene_path=prepared_root / "missing_normalized_scene.json",
        instance_registry_path=prepared_root / "missing_registry.json",
        trusted_render_source_path=prepared_root / "missing_evaluation.blend",
        consistency_report_path=consistency_path,
        readiness_report_path=readiness_path,
        provenance_path=provenance_path,
        hashes={},
    )
    manifest_path = tmp_path / "case_bundle.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    bundle = TrustedCaseBundle(
        root=tmp_path,
        manifest_path=manifest_path,
        manifest_sha256="c" * 64,
        case_id="blocked_case",
        evaluator_output_type=O3_SCENE_PACKAGE,
        scene_request={
            "request_id": "blocked_request",
            "instruction": "Place one frozen object.",
            "scene_type": "room",
            "structure": True,
            "prompt_granularity": "fine_grained",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
                "unit": "meter",
            },
        },
        reference_annotation=None,
        specification_contract=None,
        specification_activation_mode="none",
        functional_semantic_config=None,
        scene_quality_config=None,
        object_grouping_report=None,
        asset_policy=None,
        authorized_deviations=None,
        spatial_fidelity_ontology=None,
        visual_style_spec=None,
        evaluation_profile=resolve_evaluation_profile(),
        workflow="canonical_l0_l4",
        enabled_evaluators={},
        p0b_official_mode=True,
        camera_evidence={
            "mode": None,
            "metric_modes": {},
            "max_views": 2,
            "max_steps": 0,
            "collision_overlay": True,
            "collision_contour": True,
            "active_fallback": {"enabled": False},
        },
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        artifact_records={},
    )

    report = evaluate_prepared_submission(
        prepared_submission=prepared,
        case_bundle=bundle,
        out_dir=tmp_path / "evaluation",
        renderer=_ForbiddenRenderer(),
        vlm_judge=_ForbiddenJudge(),
        official_mode=False,
    )

    assert calls == {
        "renderer": 0,
        "evidence_gate": 0,
        "judge": 0,
        "metrics": 0,
    }
    assert report["evaluation_status"] == "not_evaluable"
    assert report["benchmark_score"] is None
    assert report["benchmark_score_status"] == "not_evaluable"
    assert report["layer_reports"][L0]["status"] == "not_evaluable"
    assert not (tmp_path / "evaluation" / "renders").exists()
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)


def test_official_readiness_failure_writes_audit_artifacts_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, _, _ = _catalog_bound_stub(tmp_path)
    monkeypatch.setattr(
        submission_module,
        "verify_prepared_submission",
        lambda *args, **kwargs: _failed_readiness(),
    )
    destination = tmp_path / "official_evaluation"

    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="official submission is not evaluable.*evaluation_report.json",
    ):
        evaluate_prepared_submission(
            prepared_submission=prepared,
            case_bundle=_trusted_bundle(tmp_path),
            out_dir=destination,
            official_mode=True,
        )

    readiness = read_json(destination / "evaluation_readiness_report.json")
    report = read_json(destination / "evaluation_report.json")
    manifest = read_json(destination / "submission_run_manifest.json")
    assert readiness["status"] == "not_evaluable"
    assert readiness["failure_stage"] == "materialization_consistency"
    assert readiness["primary_failure_owner"] == "generator"
    assert report["evaluation_status"] == "not_evaluable"
    assert manifest["status"] == "not_evaluable"
    assert manifest["evaluation_report"] == (
        destination / "evaluation_report.json"
    ).as_posix()


@pytest.mark.parametrize("official_mode", [False, True])
def test_artifact_entrypoint_preserves_readiness_failure_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    official_mode: bool,
) -> None:
    prepared, _, _ = _catalog_bound_stub(tmp_path)
    monkeypatch.setattr(
        submission_module,
        "prepare_submission",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        submission_module,
        "verify_prepared_submission",
        lambda *args, **kwargs: _failed_readiness(),
    )
    destination = tmp_path / f"artifact_{official_mode}"
    kwargs = {
        "artifact": {"contract_version": "catalog_placement_v1"},
        "case_bundle": _trusted_bundle(tmp_path),
        "out_dir": destination,
        "asset_root": tmp_path / "assets",
        "asset_csv": tmp_path / "assets.csv",
        "blender_bin": tmp_path / "blender",
        "official_mode": official_mode,
    }

    if official_mode:
        with pytest.raises(
            submission_module.SubmissionEvaluationError,
            match="official submission is not evaluable",
        ):
            submission_module.evaluate_artifact_submission(**kwargs)
    else:
        report = submission_module.evaluate_artifact_submission(**kwargs)
        assert report["evaluation_status"] == "not_evaluable"

    assert read_json(destination / "evaluation_report.json")[
        "evaluation_status"
    ] == "not_evaluable"
    assert read_json(destination / "submission_run_manifest.json")[
        "official_mode"
    ] is official_mode


def test_prepared_renderer_auxiliary_methods_force_and_bind_trusted_blend(
    tmp_path: Path,
) -> None:
    trusted_blend = tmp_path / "evaluation.blend"
    trusted_blend.write_bytes(b"trusted")
    expected_hash = "1" * 64
    prepared = MaterializationResult(
        normalized_scene_path=tmp_path / "normalized.json",
        instance_registry_path=tmp_path / "registry.json",
        trusted_render_source_path=trusted_blend,
        consistency_report_path=tmp_path / "consistency.json",
        readiness_report_path=tmp_path / "readiness.json",
        provenance_path=tmp_path / "provenance.json",
        hashes={"trusted_render_source_sha256": expected_hash},
    )
    captured: dict[str, Any] = {}

    class _AuxiliaryRenderer:
        def render_camera_views(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "views": [],
                "camera_evidence": {
                    "source_blend": Path(kwargs["blend_file"]).as_posix(),
                    "source_blend_sha256_before": expected_hash,
                    "source_blend_sha256_after": expected_hash,
                    "source_blend_modified": False,
                },
            }

    wrapped = submission_module._PreparedRendererAdapter(
        renderer=_AuxiliaryRenderer(),
        prepared=prepared,
    )
    result = wrapped.render_camera_views(
        blend_file=tmp_path / "submitted.blend",
        out_dir=tmp_path / "camera",
    )

    assert Path(captured["blend_file"]).resolve() == trusted_blend.resolve()
    assert (
        result["camera_evidence"]["source_blend_sha256_before"]
        == expected_hash
    )

    class _WrongHashRenderer(_AuxiliaryRenderer):
        def render_camera_views(self, **kwargs: Any) -> dict[str, Any]:
            result = super().render_camera_views(**kwargs)
            # Matching legacy root fields must not mask bad authoritative
            # nested evidence from the real auxiliary-renderer shape.
            result.update(
                {
                    "source_blend": Path(kwargs["blend_file"]).as_posix(),
                    "source_blend_sha256_before": expected_hash,
                    "source_blend_sha256_after": expected_hash,
                    "source_blend_modified": False,
                }
            )
            result["camera_evidence"]["source_blend_sha256_after"] = "2" * 64
            return result

    wrong = submission_module._PreparedRendererAdapter(
        renderer=_WrongHashRenderer(),
        prepared=prepared,
    )
    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="different trusted blend hash",
    ):
        wrong.render_camera_views(
            blend_file=trusted_blend,
            out_dir=tmp_path / "camera_wrong",
        )

    class _WrongSourceRenderer(_AuxiliaryRenderer):
        def render_camera_views(self, **kwargs: Any) -> dict[str, Any]:
            result = super().render_camera_views(**kwargs)
            result["camera_evidence"]["source_blend"] = (
                tmp_path / "submitted.blend"
            ).as_posix()
            return result

    wrong_source = submission_module._PreparedRendererAdapter(
        renderer=_WrongSourceRenderer(),
        prepared=prepared,
    )
    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="non-trusted blend source",
    ):
        wrong_source.render_camera_views(
            blend_file=trusted_blend,
            out_dir=tmp_path / "camera_wrong_source",
        )


def test_prepared_overview_renderer_binds_normalized_scene_hash(
    tmp_path: Path,
) -> None:
    trusted_blend = tmp_path / "evaluation.blend"
    trusted_blend.write_bytes(b"trusted")
    normalized = write_json(
        tmp_path / "normalized_scene.json",
        {"scene_id": "trusted", "objects": []},
    )
    blend_hash = sha256_file(trusted_blend)
    normalized_hash = sha256_file(normalized)
    prepared = MaterializationResult(
        normalized_scene_path=normalized,
        instance_registry_path=tmp_path / "registry.json",
        trusted_render_source_path=trusted_blend,
        consistency_report_path=tmp_path / "consistency.json",
        readiness_report_path=tmp_path / "readiness.json",
        provenance_path=tmp_path / "provenance.json",
        hashes={
            "trusted_render_source_sha256": blend_hash,
            "normalized_scene_sha256": normalized_hash,
        },
    )

    class _Renderer:
        def render_prepared_scene(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "blend_file": Path(kwargs["blend_file"]).as_posix(),
                "source_blend_sha256_before": blend_hash,
                "source_blend_sha256_after": blend_hash,
                "source_blend_modified": False,
                "normalized_scene_path": Path(
                    kwargs["normalized_scene_path"]
                ).as_posix(),
                "normalized_scene_sha256_before": normalized_hash,
                "normalized_scene_sha256_after": "0" * 64,
                "normalized_scene_modified": False,
            }

    wrapped = submission_module._PreparedRendererAdapter(
        renderer=_Renderer(),
        prepared=prepared,
    )
    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="different normalized scene hash",
    ):
        wrapped.render_scene(
            scene_path=normalized,
            out_dir=tmp_path / "render",
        )


def test_trusted_render_manifest_artifact_resolves_prepared_backend(
    tmp_path: Path,
) -> None:
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    prepared_manifest = write_json(
        render_dir / "prepared_render_manifest.json",
        {"backend": "blender_prepared_scene_read_only_v1"},
    )

    resolved = submission_module._trusted_render_manifest_artifact(
        {"backend": "blender_prepared_scene_read_only_v1"},
        render_dir,
    )

    assert resolved == prepared_manifest.resolve()
    assert resolved.name == "prepared_render_manifest.json"
    assert submission_module._trusted_render_manifest_artifact(
        {
            "backend": "blender_prepared_scene_read_only_v1",
            "manifest_path": (tmp_path / "outside.json").as_posix(),
        },
        render_dir,
    ) == prepared_manifest.resolve()


def test_prepared_incomplete_official_report_is_finalized_before_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    normalized = write_json(prepared_root / "normalized_scene.json", {})
    authority = tmp_path / "evaluation" / "fresh" / "evaluation.blend"
    authority.parent.mkdir(parents=True)
    authority.write_bytes(b"fresh-frozen-authority")
    authority_hash = sha256_file(authority)
    prepared = MaterializationResult(
        normalized_scene_path=normalized,
        instance_registry_path=prepared_root / "registry.json",
        trusted_render_source_path=prepared_root / "evaluation.blend",
        consistency_report_path=prepared_root / "consistency.json",
        readiness_report_path=prepared_root / "readiness.json",
        provenance_path=prepared_root / "provenance.json",
        hashes={"normalized_scene_sha256": sha256_file(normalized)},
    )
    ready = build_readiness_report(
        status="ready",
        checks=[{"id": "prepared_artifact_integrity", "passed": True}],
        provenance={
            "evaluation_time_trust_audit": {
                "frozen_authority_blend_path": authority.as_posix(),
                "frozen_authority_blend_sha256": authority_hash,
            }
        },
    )
    monkeypatch.setattr(
        submission_module,
        "verify_prepared_submission",
        lambda *args, **kwargs: deepcopy(ready),
    )
    monkeypatch.setattr(
        submission_module,
        "_audit_prepared_submission_for_evaluation",
        lambda **kwargs: (
            deepcopy(ready),
            tmp_path / "assets",
            tmp_path / "assets.csv",
        ),
    )
    captured: dict[str, Any] = {}

    def incomplete_evaluation(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        destination = Path(kwargs["out_dir"])
        preliminary_report = {
            "evaluation_status": "incomplete",
            "benchmark_score": None,
            "benchmark_score_status": "insufficient_metric_coverage",
            "layer_reports": {
                L0: {
                    "layer": L0,
                    "status": "passed",
                    "score": None,
                    "affects_score": False,
                }
            },
            "category_reports": {
                L0: {
                    "layer": L0,
                    "status": "passed",
                    "score": None,
                    "affects_score": False,
                }
            },
            "evidence_provenance": {
                "render_input_policy": "preliminary",
            },
        }
        preliminary_manifest = {
            "status": "incomplete",
            "rendering": {
                "input_policy": "preliminary",
                "input_path": "preliminary",
                "manifest_path": None,
                "manifest_sha256": None,
                "overview_views": [],
            },
        }
        write_json(
            destination / "submission_run_manifest.json",
            preliminary_manifest,
        )
        return {
            "evaluation_report": preliminary_report,
            "manifest": preliminary_manifest,
        }

    monkeypatch.setattr(
        submission_module,
        "_evaluate_submission_impl",
        incomplete_evaluation,
    )

    class _Renderer:
        def render_prepared_scene(self, **_: Any) -> dict[str, Any]:
            raise AssertionError("mocked evaluation must own rendering")

    evaluation_dir = tmp_path / "evaluation"
    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="complete metric coverage",
    ):
        evaluate_prepared_submission(
            prepared_submission=prepared,
            case_bundle=_trusted_bundle(tmp_path),
            out_dir=evaluation_dir,
            renderer=_Renderer(),
            vlm_judge=object(),
            official_mode=True,
        )

    assert captured["defer_incomplete_error"] is True
    persisted_report = read_json(evaluation_dir / "evaluation_report.json")
    assert persisted_report["layer_reports"][L0]["readiness"]["status"] == "ready"
    assert persisted_report["evidence_provenance"][
        "trusted_render_source"
    ] == authority.as_posix()
    assert persisted_report["evidence_provenance"][
        "trusted_render_source_rederived_at_evaluation"
    ] is True
    persisted_manifest = read_json(
        evaluation_dir / "submission_run_manifest.json"
    )
    assert persisted_manifest["evaluation_render_authority"] == {
        "source": "fresh_frozen_catalog_rematerialization",
        "path": authority.as_posix(),
        "sha256": authority_hash,
    }
    assert persisted_manifest["rendering"]["input_path"] == authority.as_posix()
    assert persisted_manifest["evaluation_report"] == (
        evaluation_dir / "evaluation_report.json"
    ).as_posix()


def test_prepared_csv_binding_mismatch_is_readiness_failure_before_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "inspection": 0,
        "renderer": 0,
        "evidence_gate": 0,
        "judge": 0,
        "metrics": 0,
    }

    class _ForbiddenRenderer:
        blender_bin = tmp_path / "blender"

        def render_prepared_scene(self, **_: Any) -> dict:
            calls["renderer"] += 1
            raise AssertionError("renderer must not run after catalog mismatch")

    class _ForbiddenJudge:
        def __getattr__(self, _: str):
            def forbidden(*args: Any, **kwargs: Any) -> dict:
                del args, kwargs
                calls["judge"] += 1
                raise AssertionError("judge must not run after catalog mismatch")

            return forbidden

    def forbidden_inspection(**_: Any) -> dict:
        calls["inspection"] += 1
        raise AssertionError("blend inspection must not run after catalog mismatch")

    def forbidden_metrics(*args: Any, **kwargs: Any) -> dict:
        del args, kwargs
        calls["metrics"] += 1
        raise AssertionError("metrics must not run after catalog mismatch")

    def forbidden_evidence_gate(*args: Any, **kwargs: Any):
        del args, kwargs
        calls["evidence_gate"] += 1
        raise AssertionError("EvidenceGate must not run after catalog mismatch")

    prepared, asset_root, prepared_csv = _catalog_bound_stub(tmp_path)
    replacement_csv = tmp_path / "replacement_assets.csv"
    replacement_csv.write_text("name_en\nasset\n", encoding="utf-8")
    verified_ready = build_readiness_report(
        status="ready",
        checks={"prepared_artifact_integrity": True},
        provenance={},
    )
    monkeypatch.setattr(
        submission_module,
        "verify_prepared_submission",
        lambda *args, **kwargs: deepcopy(verified_ready),
    )
    monkeypatch.setattr(
        submission_module,
        "inspect_sanitized_blend",
        forbidden_inspection,
    )
    monkeypatch.setattr(submission_module, "run_evaluate", forbidden_metrics)
    monkeypatch.setattr(
        DeterministicEvidenceGate,
        "check",
        forbidden_evidence_gate,
    )

    report = evaluate_prepared_submission(
        prepared_submission=prepared,
        case_bundle=_trusted_bundle(tmp_path),
        out_dir=tmp_path / "evaluation",
        renderer=_ForbiddenRenderer(),
        vlm_judge=_ForbiddenJudge(),
        asset_root=asset_root,
        asset_csv=replacement_csv,
        official_mode=False,
    )

    assert calls == {
        "inspection": 0,
        "renderer": 0,
        "evidence_gate": 0,
        "judge": 0,
        "metrics": 0,
    }
    assert report["evaluation_status"] == "not_evaluable"
    assert report["benchmark_score"] is None
    readiness = report["layer_reports"][L0]["readiness"]
    assert readiness["status"] == "not_evaluable"
    assert "catalog_asset_csv_binding_mismatch" in readiness["reason_codes"]
    assert not (tmp_path / "evaluation" / "renders").exists()


def test_official_prepared_catalog_arguments_are_required_before_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _ForbiddenRenderer:
        blender_bin = tmp_path / "blender"

        def render_prepared_scene(self, **_: Any) -> dict:
            calls.append("renderer")
            raise AssertionError("renderer must not run without trusted catalog args")

    class _ForbiddenJudge:
        def __getattr__(self, _: str):
            def forbidden(*args: Any, **kwargs: Any) -> dict:
                del args, kwargs
                calls.append("judge")
                raise AssertionError("judge must not run without trusted catalog args")

            return forbidden

    def forbidden_inspection(**_: Any) -> dict:
        calls.append("inspection")
        raise AssertionError("inspection must not run without trusted catalog args")

    def forbidden_metrics(*args: Any, **kwargs: Any) -> dict:
        del args, kwargs
        calls.append("metrics")
        raise AssertionError("metrics must not run without trusted catalog args")

    def forbidden_evidence_gate(*args: Any, **kwargs: Any):
        del args, kwargs
        calls.append("evidence_gate")
        raise AssertionError(
            "EvidenceGate must not run without trusted catalog args"
        )

    prepared, _, _ = _catalog_bound_stub(tmp_path)
    verified_ready = build_readiness_report(
        status="ready",
        checks={"prepared_artifact_integrity": True},
        provenance={},
    )
    monkeypatch.setattr(
        submission_module,
        "verify_prepared_submission",
        lambda *args, **kwargs: deepcopy(verified_ready),
    )
    monkeypatch.setattr(
        submission_module,
        "inspect_sanitized_blend",
        forbidden_inspection,
    )
    monkeypatch.setattr(submission_module, "run_evaluate", forbidden_metrics)
    monkeypatch.setattr(
        DeterministicEvidenceGate,
        "check",
        forbidden_evidence_gate,
    )

    destination = tmp_path / "evaluation"
    with pytest.raises(
        submission_module.SubmissionEvaluationError,
        match="official submission is not evaluable.*evaluation_report.json",
    ):
        evaluate_prepared_submission(
            prepared_submission=prepared,
            case_bundle=_trusted_bundle(tmp_path),
            out_dir=destination,
            renderer=_ForbiddenRenderer(),
            vlm_judge=_ForbiddenJudge(),
            official_mode=True,
        )

    assert calls == []
    report = read_json(destination / "evaluation_report.json")
    assert report["evaluation_status"] == "not_evaluable"
    assert report["benchmark_score"] is None
    readiness = report["layer_reports"][L0]["readiness"]
    assert readiness["status"] == "not_evaluable"
    assert {
        "official_catalog_asset_root_required",
        "official_catalog_asset_csv_required",
    }.issubset(readiness["reason_codes"])
    assert readiness["failure_stage"] == "evaluation_time_trust_audit"
    assert readiness["primary_failure_owner"] == "submission"
    manifest = read_json(destination / "submission_run_manifest.json")
    assert manifest["status"] == "not_evaluable"
    assert not (tmp_path / "evaluation" / "renders").exists()


def test_evaluation_audit_derives_catalog_and_reruns_fresh_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _trusted_bundle(tmp_path)
    generation_input = _trusted_generation_input()
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "asset"
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset.obj").write_text(
        "o asset\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,A chair,A chair,"[1, 1, 1]"\n',
        encoding="utf-8",
    )
    catalog = FrozenCatalog(
        asset_csv=asset_csv,
        asset_root=asset_root,
        allowed_asset_ids=bundle.allowed_asset_ids,
        snapshot_id=str(bundle.catalog_snapshot_id),
    )
    plan = preparation_module._build_plan(
        {
            "instances": [
                {
                    "instance_id": "chair",
                    "asset_id": "asset",
                        "center_m": [1.0, 1.0, 0.5],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                        "slot_id": "chair_slot",
                }
            ]
        },
            case_bundle=bundle,
            catalog=catalog,
            task_slots=preparation_module._task_slots_for_generation_input(
                generation_input
            ),
        )
    expected = plan["instances"][0]
    blend_inspection = {
        "status": "passed",
        "instances": [
            {
                "instance_id": expected["instance_id"],
                "evaluator_object_id": expected["evaluator_object_id"],
                "asset_id": expected["asset_id"],
                "slot_id": expected["slot_id"],
                "center_m": deepcopy(expected["center_m"]),
                "requested_uniform_scale": expected[
                    "requested_uniform_scale"
                ],
                "effective_uniform_scale": expected[
                    "effective_uniform_scale"
                ],
                "actual_local_bbox_size_m": deepcopy(
                    expected["actual_local_bbox_size_m"]
                ),
                "rotation_euler_xyz_deg": deepcopy(
                    expected["rotation_euler_xyz_deg"]
                ),
                "uniform_scale": expected["uniform_scale"],
                "local_bbox_size_m": deepcopy(expected["local_bbox_size_m"]),
                "world_bounds": deepcopy(expected["world_bounds"]),
                    "geometry_sha256": "d" * 64,
                    "asset_assembly_sha256": "f" * 64,
                    "material_sha256": "e" * 64,
                "root_object_name": "benchmark_instance_chair",
                "render_enabled": True,
            }
        ],
        "technical_state": {
            "all_instances_render_enabled": True,
            "extra_renderable_instance_count": 0,
        },
    }
    normalized_scene, instance_registry = (
        preparation_module._export_scene_and_registry(plan, blend_inspection)
    )

    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    raw_source_path = write_json(
        prepared_root / "raw_generator_artifact.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair",
                    "asset_id": "asset",
                    "center_m": [1.0, 1.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_slot",
                }
            ],
        },
    )
    plan_path = write_json(prepared_root / "materialization_plan.json", plan)
    normalized_path = write_json(
        prepared_root / "normalized_scene.json",
        normalized_scene,
    )
    registry_path = write_json(
        prepared_root / "instance_registry.json",
        instance_registry,
    )
    trusted_path = prepared_root / "evaluation.blend"
    trusted_path.write_bytes(b"trusted blend")
    trusted_hash = sha256_file(trusted_path)
    prepared_hashes = {
        "source_artifact_sha256": sha256_file(raw_source_path),
        "normalized_scene_sha256": sha256_file(normalized_path),
        "instance_registry_sha256": sha256_file(registry_path),
        "trusted_render_source_sha256": trusted_hash,
        "materialization_plan_sha256": sha256_file(plan_path),
        "trusted_blend_inspection_sha256": "2" * 64,
        "adapter_contract_revision_sha256": hashlib.sha256(
            b"catalog_placement_v1"
        ).hexdigest(),
        "catalog_csv_sha256": catalog.catalog_csv_sha256,
        "generator_visible_input_sha256": sha256_json(
            generation_input
        ),
    }
    provenance_core_path = write_json(
        prepared_root / "provenance_core.json",
        {
            "provenance_core_version": (
                "catalog_materialization_provenance_core_v1"
            ),
            "adapter_contract_revision": "catalog_placement_v1",
            "materialization_revision": "fixed_catalog_materialization_v1",
            "case_id": bundle.case_id,
            "case_bundle_manifest_sha256": bundle.manifest_sha256,
            "catalog_snapshot_id": catalog.snapshot_id,
            "catalog_csv_path": catalog.asset_csv.as_posix(),
            "asset_root_path": catalog.asset_root.as_posix(),
            "source": {
                "kind": "in_memory_json",
                "preserved_path": raw_source_path.as_posix(),
                "sha256": sha256_file(raw_source_path),
            },
            "generator_visible_input": {
                "sha256": sha256_json(generation_input),
                "request_id": "blocked_request",
                "selected_asset_ids": ["asset"],
            },
            "representation_hashes": deepcopy(prepared_hashes),
        },
    )
    prepared_hashes["provenance_core_sha256"] = sha256_file(
        provenance_core_path
    )
    provenance_path = write_json(
        prepared_root / "provenance.json",
        {
            "source": {
                "kind": "in_memory_json",
                "preserved_path": raw_source_path.as_posix(),
                "sha256": sha256_file(raw_source_path),
            },
            "generator_visible_input": {
                "sha256": sha256_json(generation_input),
                "request_id": "blocked_request",
                "selected_asset_ids": ["asset"],
            },
            "catalog": {
                "snapshot_id": catalog.snapshot_id,
                "asset_csv_path": catalog.asset_csv.as_posix(),
                "asset_root_path": catalog.asset_root.as_posix(),
                "catalog_csv_sha256": catalog.catalog_csv_sha256,
            },
            "artifacts": {
                "materialization_plan": plan_path.as_posix(),
                "provenance_core": provenance_core_path.as_posix(),
            },
        },
    )
    prepared = MaterializationResult(
        normalized_scene_path=normalized_path,
        instance_registry_path=registry_path,
        trusted_render_source_path=trusted_path,
        consistency_report_path=prepared_root / "consistency_report.json",
        readiness_report_path=prepared_root / "readiness_report.json",
        provenance_path=provenance_path,
        hashes=prepared_hashes,
    )
    observed: dict[str, Any] = {}

    def fake_inspection(**kwargs: Any) -> dict:
        observed.update(kwargs)
        result = deepcopy(blend_inspection)
        result["source_integrity"] = {
            "source_blend_sha256_before": trusted_hash,
            "source_blend_sha256_after": trusted_hash,
            "source_blend_modified": False,
        }
        write_json(Path(kwargs["out_path"]), result)
        return result

    def fake_frozen_materialization(**kwargs: Any) -> dict:
        Path(kwargs["out_blend_path"]).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(kwargs["out_blend_path"]).write_bytes(b"fresh frozen blend")
        result = deepcopy(blend_inspection)
        write_json(Path(kwargs["inspection_path"]), result)
        return result

    monkeypatch.setattr(
        submission_module,
        "inspect_sanitized_blend",
        fake_inspection,
    )
    monkeypatch.setattr(
        submission_module,
        "materialize_catalog_scene",
        fake_frozen_materialization,
    )
    destination = tmp_path / "evaluation"
    destination.mkdir()
    base_readiness = build_readiness_report(
        status="ready",
        checks={"prepared_artifact_integrity": True},
        provenance={},
    )

    audited, resolved_root, resolved_csv = (
        submission_module._audit_prepared_submission_for_evaluation(
            prepared=prepared,
            bundle=bundle,
            destination=destination,
            readiness=base_readiness,
            renderer=type(
                "_Renderer",
                (),
                {"blender_bin": tmp_path / "blender", "timeout_seconds": 7},
            )(),
            asset_root=None,
                asset_csv=None,
                blender_bin=None,
                generation_input=generation_input,
                official_mode=False,
        )
    )

    assert audited["status"] == "ready"
    assert audited["checks"][-1]["id"] == "evaluation_time_trust_audit"
    assert audited["checks"][-1]["passed"] is True
    assert resolved_root == asset_root.resolve()
    assert resolved_csv == asset_csv.resolve()
    assert observed["blend_path"] == trusted_path
    assert observed["expected_registry_path"] == plan_path.resolve()
    assert observed["timeout_seconds"] == 7
    fresh_consistency = read_json(
        destination / "evaluation_materialization_consistency.json"
    )
    assert fresh_consistency["status"] == "passed"

    original_scene = read_json(normalized_path)
    forged_scene = deepcopy(original_scene)
    forged_scene["relations"] = [
        {
            "subject_id": "chair",
            "type": "near",
            "object_id": "chair",
        }
    ]
    write_json(normalized_path, forged_scene)
    prepared.hashes["normalized_scene_sha256"] = sha256_file(
        normalized_path
    )
    relation_core = read_json(provenance_core_path)
    relation_core["representation_hashes"][
        "normalized_scene_sha256"
    ] = prepared.hashes["normalized_scene_sha256"]
    write_json(provenance_core_path, relation_core)
    prepared.hashes["provenance_core_sha256"] = sha256_file(
        provenance_core_path
    )
    relation_destination = tmp_path / "relation_forged_evaluation"
    relation_destination.mkdir()

    relation_rejected, _, _ = (
        submission_module._audit_prepared_submission_for_evaluation(
            prepared=prepared,
            bundle=bundle,
            destination=relation_destination,
            readiness=base_readiness,
            renderer=type(
                "_Renderer",
                (),
                {"blender_bin": tmp_path / "blender", "timeout_seconds": 7},
            )(),
            asset_root=None,
            asset_csv=None,
            blender_bin=None,
            generation_input=generation_input,
            official_mode=False,
        )
    )

    assert relation_rejected["status"] == "not_evaluable"
    assert "deterministic_normalized_scene_mismatch" in (
        relation_rejected["reason_codes"]
    )

    write_json(normalized_path, original_scene)
    prepared.hashes["normalized_scene_sha256"] = sha256_file(
        normalized_path
    )
    restored_core = read_json(provenance_core_path)
    restored_core["representation_hashes"][
        "normalized_scene_sha256"
    ] = prepared.hashes["normalized_scene_sha256"]
    write_json(provenance_core_path, restored_core)
    prepared.hashes["provenance_core_sha256"] = sha256_file(
        provenance_core_path
    )

    write_json(
        raw_source_path,
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair",
                    "asset_id": "asset",
                    "center_m": [2.0, 1.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_slot",
                }
            ],
        },
    )
    forged_source_hash = sha256_file(raw_source_path)
    prepared.hashes["source_artifact_sha256"] = forged_source_hash
    forged_core = read_json(provenance_core_path)
    forged_core["source"]["sha256"] = forged_source_hash
    forged_core["representation_hashes"][
        "source_artifact_sha256"
    ] = forged_source_hash
    write_json(provenance_core_path, forged_core)
    prepared.hashes["provenance_core_sha256"] = sha256_file(
        provenance_core_path
    )
    forged_provenance = read_json(provenance_path)
    forged_provenance["source"]["sha256"] = forged_source_hash
    write_json(provenance_path, forged_provenance)
    observed.clear()
    forged_destination = tmp_path / "forged_evaluation"
    forged_destination.mkdir()

    rejected, rejected_root, rejected_csv = (
        submission_module._audit_prepared_submission_for_evaluation(
            prepared=prepared,
            bundle=bundle,
            destination=forged_destination,
            readiness=base_readiness,
            renderer=type(
                "_Renderer",
                (),
                {"blender_bin": tmp_path / "blender", "timeout_seconds": 7},
            )(),
            asset_root=None,
            asset_csv=None,
            blender_bin=None,
            generation_input=generation_input,
            official_mode=False,
        )
    )

    assert rejected["status"] == "not_evaluable"
    assert "generator_source_plan_binding_mismatch" in (
        rejected["reason_codes"]
    )
    assert rejected_root is None
    assert rejected_csv is None
    assert observed == {}
