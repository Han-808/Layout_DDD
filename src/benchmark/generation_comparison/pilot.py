"""Prepare, execute, and aggregate the first controlled generation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    load_asset_catalog,
)
from benchmark.adapters.common.execution import (
    bridge_bundle_identity,
    redact_private_locators,
)
from benchmark.generation_comparison.eligibility import check_method_eligibility
from benchmark.generation_comparison.execution import run_controlled_generation
from benchmark.generation_comparison.evaluation_runtime import (
    CanonicalEvaluationRuntime,
    evaluator_policy_readiness,
)
from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.generation_comparison.imaginarium_bundle import (
    bundle_mesh_path,
    validate_imaginarium_glb_bundle,
)
from benchmark.generation_comparison.model_policy import (
    api_base_sha256,
    configured_model_policy_report,
)
from benchmark.generation_comparison.protocol import ComparisonProtocol
from benchmark.generation_comparison.prepared import (
    freeze_prepared_artifacts,
    verify_prepared_artifacts,
)
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
    build_generation_input,
    build_scene_request,
)
from benchmark.scene_io.validate import ArtifactValidationError, validate_object_plan
from benchmark.utils.io import read_json, write_json


PILOT_SCHEMA_VERSION = "controlled_generation_pilot_v1"
PILOT_MANIFEST_SCHEMA_VERSION = "controlled_generation_pilot_manifest_v1"
ASSET_PREFLIGHT_SCHEMA_VERSION = "controlled_asset_preflight_v1"
RESULT_SCHEMA_VERSION = "controlled_generation_pilot_result_v1"
ASSET_SELECTION_PENDING = "candidate_pending_human_approval"
ASSET_SELECTION_APPROVED = "human_approved"
FAILURE_CLASSES = {
    "generation_failure",
    "timeout",
    "invalid_native_artifact",
    "protocol_violation",
    "asset_identity_violation",
    "architecture_violation",
    "canonicalization_failure",
    "evaluator_infrastructure_failure",
}
RESULT_COLUMNS = (
    "case_id",
    "method",
    "run_status",
    "attempted",
    "execution_mode",
    "upstream_process_started",
    "protocol_id",
    "protocol_hash",
    "architecture_hash",
    "inventory_hash",
    "asset_binding_hash",
    "evaluator_config_hash",
    "seed",
    "seed_enforcement",
    "object_count",
    "room_area",
    "object_density",
    "pairwise_interaction_proxy",
    "generation_success",
    "valid_comparison_run",
    "evaluation_success",
    "score_available",
    "benchmark_score",
    "benchmark_score_100",
    "benchmark_score_status",
    "layer_scores",
    "hard_metric_outcomes",
    "hard_failure_count",
    "generation_runtime_seconds",
    "generation_iterations",
    "tokens",
    "tool_calls",
    "failure_class",
    "failure_source",
    "failure_reason",
    "run_manifest",
    "evaluation_report",
    "initial_score",
    "final_score",
    "score_delta",
    "iteration_count",
    "success_at_iteration",
    "trajectory_regression_count",
    "trajectory_hard_failure_fixes",
    "trajectory_hard_failure_regressions",
)


def prepare_controlled_pilot(
    *,
    spec: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
    out_dir: str | Path,
    method_configs: Mapping[str, Any] | str | Path | None = None,
    repo_root: str | Path | None = None,
    asset_bundle_root: str | Path | None = None,
    evaluation_runtime_config: Mapping[str, Any] | str | Path | None = None,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze inputs without generation, optionally selecting existing cases.

    A subset is selected from the fully validated source spec in source order.
    It does not change any case, shrink the catalog, or select different cases
    for different methods. Each stage/repetition still needs a fresh directory.
    """

    pilot = _load_mapping(spec, "pilot spec")
    _validate_pilot_spec(pilot)
    source_case_ids = [str(case["case_id"]) for case in pilot["cases"]]
    if case_ids is not None:
        if (
            isinstance(case_ids, (str, bytes)) or not case_ids
            or any(not isinstance(value, str) for value in case_ids)
            or len(set(case_ids)) != len(case_ids)
            or not set(case_ids).issubset(source_case_ids)
        ):
            raise ArtifactValidationError(
                "case_ids must be a non-empty unique subset of the source case IDs"
            )
    selected_cases = [
        case for case in pilot["cases"]
        if case_ids is None or case["case_id"] in case_ids
    ]
    case_selection = {
        "policy": "all_source_cases" if case_ids is None else "explicit_subset_source_order",
        "source_case_ids": source_case_ids,
        "selected_case_ids": [str(case["case_id"]) for case in selected_cases],
        "source_spec_sha256": canonical_json_sha256(pilot),
        "case_definitions_modified": False,
        "catalog_subsetted": False,
    }
    output_root = Path(out_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"pilot output already exists and will not be overwritten: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(asset_root).expanduser().resolve()
    catalog, asset_preflight = _preflight_imaginarium_catalog(
        pilot["catalog"],
        source_root,
        asset_bundle_root=(
            Path(asset_bundle_root).expanduser().resolve()
            if asset_bundle_root is not None
            else None
        ),
    )
    asset_preflight_path = write_json(
        output_root / "asset_preflight.json", asset_preflight
    )
    if asset_preflight["status"] != "passed":
        write_json(output_root / "catalog_manifest.invalid.json", catalog.as_dict())
        raise ArtifactValidationError(
            "frozen asset preflight failed; generation was not launched"
        )
    catalog_path = write_json(output_root / "catalog_manifest.json", catalog.as_dict())

    evaluator_policy = dict(pilot["evaluator"])
    if evaluation_runtime_config is not None:
        evaluator_policy["runtime"] = _load_mapping(evaluation_runtime_config, "evaluator runtime config")
        # Validate without contacting any service, and reject literal credentials
        # before the private runtime configuration is written to disk.
        CanonicalEvaluationRuntime(evaluator_policy["runtime"], require_credentials=False)
        # Resolve existing defaults at prepare time, not differently in each run.
        from benchmark.evaluator.profile import resolve_evaluation_profile
        static = dict(evaluator_policy.get("static_kwargs") or {})
        static["evaluation_profile"] = resolve_evaluation_profile(static.get("evaluation_profile"))
        frozen_ownership = {
            "mode": "benchmark_provided", "identity_owner": "benchmark",
            "category_selection_owner": "benchmark", "scale_owner": "benchmark",
            "appearance_owner": "benchmark", "arrangement_owner": "generator",
        }
        if "asset_policy" in static and static["asset_policy"] != frozen_ownership:
            raise ArtifactValidationError("evaluator asset ownership contradicts fixed-assets input")
        # Factual ownership activates the existing applicability rules. We do
        # not enable/disable metrics or change weights to conceal missing runtime.
        static["asset_policy"] = frozen_ownership
        evaluator_policy["static_kwargs"] = static
        evaluator_policy["rendering_policy"] = "trusted_blender_mesh_runtime_v1"
        evaluator_policy["evidence_policy"] = "canonical_judge_grouping_camera_runtime_v1"
    evaluator_config_hash = canonical_json_sha256(evaluator_policy)
    evaluator_policy["config_sha256"] = evaluator_config_hash
    evaluator_path = write_json(
        output_root / "evaluator_config.json", evaluator_policy
    )
    case_rows = []
    case_protocols: dict[str, str] = {}
    for case in selected_cases:
        case_id = str(case["case_id"])
        case_dir = output_root / "cases" / case_id
        protocol = _case_protocol(
            pilot=pilot,
            case=case,
            catalog=catalog,
            evaluator_policy=evaluator_policy,
        )
        generation_input = _generation_input_for_case(pilot=pilot, case=case)
        object_plan = _evaluation_object_plan(case)
        complexity = _case_complexity(case)
        protocol_path = write_json(case_dir / "protocol.json", protocol.as_dict())
        generation_path = write_json(
            case_dir / "generation_input.json", generation_input
        )
        object_plan_path = write_json(
            case_dir / "evaluation_object_plan.json", object_plan
        )
        case_definition = {
            "case_id": case_id,
            "scene_type": case["scene_type"],
            "seed": case.get("seed"),
            "room": case["room"],
            "instruction": case["instruction"],
            "objects": case["objects"],
            **(
                {"object_plan": case["object_plan"]}
                if isinstance(case.get("object_plan"), Mapping)
                else {}
            ),
            **(
                {"source_provenance": case["source_provenance"]}
                if isinstance(case.get("source_provenance"), Mapping)
                else {}
            ),
        }
        case_manifest = {
            "schema_version": "controlled_generation_case_manifest_v1",
            "pilot_id": pilot["pilot_id"],
            "case_id": case_id,
            "case_sha256": canonical_json_sha256(case_definition),
            "seed": case.get("seed"),
            "seed_enforcement": "not_guaranteed_unless_runner_reports",
            "protocol": protocol_path.resolve().as_posix(),
            "protocol_sha256": protocol.sha256,
            "architecture_sha256": protocol.architecture_hash,
            "object_inventory_sha256": protocol.inventory_sha256,
            "asset_binding_sha256": protocol.binding_sha256,
            "catalog": catalog.identity,
            "evaluator_config_sha256": evaluator_config_hash,
            "generation_input": generation_path.resolve().as_posix(),
            "generation_input_sha256": canonical_json_sha256(generation_input),
            "evaluation_object_plan": object_plan_path.resolve().as_posix(),
            "public_object_plan_sha256": canonical_json_sha256(object_plan),
            "source_provenance": deepcopy(case.get("source_provenance") or {}),
            "complexity": complexity,
        }
        case_manifest_path = write_json(case_dir / "case_manifest.json", case_manifest)
        case_protocols[case_id] = protocol.sha256
        case_rows.append(
            {
                **case_manifest,
                "case_manifest": case_manifest_path.resolve().as_posix(),
            }
        )

    configs = _method_configs(method_configs)
    first_protocol = ComparisonProtocol.from_mapping(
        read_json(output_root / "cases" / selected_cases[0]["case_id"] / "protocol.json")
    )
    compatibility = _compatibility_report(
        methods=pilot["methods"],
        protocol=first_protocol,
        catalog=catalog,
        method_configs=configs,
    )
    compatibility_path = write_json(
        output_root / "compatibility_report.json", compatibility
    )
    root_protocol = {
        "schema_version": "controlled_generation_pilot_protocol_manifest_v1",
        "pilot_id": pilot["pilot_id"],
        "protocol_id": pilot["protocol_id"],
        "protocol_version": pilot["protocol_version"],
        "mode": pilot["mode"],
        "asset_selection_status": pilot.get("asset_selection_status"),
        "catalog": catalog.identity,
        "evaluator_config_sha256": evaluator_config_hash,
        "case_protocol_sha256": case_protocols,
        "case_selection": case_selection,
    }
    protocol_path = write_json(output_root / "protocol.json", root_protocol)
    manifest = {
        "schema_version": PILOT_MANIFEST_SCHEMA_VERSION,
        "pilot_id": pilot["pilot_id"],
        "label": pilot.get("label") or "pilot / integration validation",
        "status": "prepared",
        "asset_selection_status": pilot.get("asset_selection_status"),
        "branch_commit": _git_commit(
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        ),
        "source_spec_sha256": canonical_json_sha256(pilot),
        "protocol": protocol_path.resolve().as_posix(),
        "catalog": catalog_path.resolve().as_posix(),
        "asset_preflight": asset_preflight_path.resolve().as_posix(),
        "evaluator_config": evaluator_path.resolve().as_posix(),
        "evaluator_config_sha256": evaluator_config_hash,
        "compatibility_report": compatibility_path.resolve().as_posix(),
        "methods": list(pilot["methods"]),
        "case_count": len(case_rows),
        "cases": case_rows,
        "real_upstream_execution_performed": False,
    }
    freeze_prepared_artifacts(output_root, manifest)
    manifest_path = write_json(output_root / "pilot_manifest.json", manifest)
    _write_readme(output_root, manifest, compatibility, summary=None)
    return {**manifest, "manifest_path": manifest_path.resolve().as_posix()}


def _evaluation_readiness(
    policy: Mapping[str, Any],
) -> tuple[CanonicalEvaluationRuntime | None, dict[str, Any]]:
    runtime = None
    report = {"ready": False, "reasons": ["evaluator_runtime_missing"],
              "service_contacted": False}
    if policy.get("runtime") is not None:
        try:
            runtime = CanonicalEvaluationRuntime(policy["runtime"])
            report = runtime.readiness
        except Exception as exc:
            report["reasons"] = [redact_private_locators(str(exc))]
    score_policy = evaluator_policy_readiness(policy)
    return runtime, {**report, "ready": report["ready"] and score_policy["ready"],
                     "reasons": list(report.get("reasons", [])) + score_policy["reasons"],
                     "score_policy": score_policy}


def preflight_prepared_pilot(
    *, prepared_dir: str | Path,
    method_configs: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Read-only, no-model/no-render preflight; does NOT call the run command."""
    root = Path(prepared_dir).expanduser().resolve()
    manifest = read_json(root / "pilot_manifest.json")
    verified = verify_prepared_artifacts(root, manifest)
    configs = _method_configs(method_configs)
    _, runtime = _evaluation_readiness(verified["evaluator_policy"])
    cases = []
    for case in manifest["cases"]:
        compatibility = _compatibility_report(
            methods=manifest["methods"],
            protocol=verified["cases"][case["case_id"]]["protocol"],
            catalog=verified["catalog"], method_configs=configs,
        )
        cases.append({"case_id": case["case_id"], **compatibility})
    methods_ready = all(row["status"] == "ELIGIBLE_READY"
                        for case in cases for row in case["methods"])
    approved = manifest.get("asset_selection_status") == ASSET_SELECTION_APPROVED
    return {
        "schema_version": "controlled_generation_no_call_preflight_v1",
        "ready_for_generation": bool(approved and methods_ready and runtime["ready"]),
        "asset_selection_approved": approved,
        "prepared_artifacts": verified["report"],
        "method_configs_sha256": canonical_json_sha256(configs),
        "evaluation_runtime": runtime, "cases": cases,
        "service_contacted": False, "generation_executed": False,
        "rendering_executed": False, "real_upstream_smoke_verified": False,
    }


def run_prepared_pilot(
    *,
    prepared_dir: str | Path,
    method_configs: Mapping[str, Any] | str | Path | None = None,
    method_outputs: Mapping[str, Mapping[str, str | Path]] | None = None,
    dry_run_only: bool = False,
    allow_offline_artifacts: bool = False,
) -> dict[str, Any]:
    """Run dry case first, then remaining cases for methods that pass it."""

    root = Path(prepared_dir).expanduser().resolve()
    manifest_path = root / "pilot_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ArtifactValidationError("prepared pilot manifest is invalid")
    asset_selection_status = manifest.get("asset_selection_status")
    if (
        asset_selection_status is not None
        and asset_selection_status != ASSET_SELECTION_APPROVED
    ):
        raise ArtifactValidationError(
            "frozen asset selection is not approved for generation: "
            f"{asset_selection_status!r}; review the candidate bindings, set "
            f"asset_selection_status={ASSET_SELECTION_APPROVED!r}, and prepare "
            "a fresh immutable pilot directory"
        )
    if (root / "results.jsonl").exists() or (root / "results.csv").exists():
        raise FileExistsError(
            "pilot result tables already exist; failed/completed runs are never overwritten"
        )
    if manifest.get("status") != "prepared":
        raise ArtifactValidationError("pilot is not a fresh prepared run")
    verified = verify_prepared_artifacts(root, manifest)
    write_json(root / "run_preflight.json", verified["report"])
    catalog = verified["catalog"]
    evaluation_runtime, runtime_readiness = _evaluation_readiness(verified["evaluator_policy"])
    write_json(root / "evaluation_readiness.json", runtime_readiness)
    configs = _method_configs(method_configs)
    outputs = method_outputs or {}
    cases = list(manifest["cases"])
    if not cases:
        raise ArtifactValidationError("prepared pilot has no cases")

    rows: list[dict[str, Any]] = []
    passed_dry_run: list[str] = []
    planned_cases = cases[:1] if dry_run_only else cases
    plan = [(case, method) for case in planned_cases for method in manifest["methods"]]
    for index, (case, method) in enumerate(plan):
        offline = outputs.get(method, {}).get(case["case_id"])
        # Validate all prepared cases at the start, and each unit again before
        # launching; edits to a later case must not spend any generation calls.
        protocol = verified["cases"][case["case_id"]]["protocol"]
        active_model_policy = protocol.as_dict()["generation"].get("model_policy")
        active_model_policy = (
            active_model_policy if isinstance(active_model_policy, Mapping) else {}
        )
        active_harness_inputs = protocol.as_dict()["generation"].get(
            "harness_inputs"
        )
        active_harness_inputs = (
            active_harness_inputs
            if isinstance(active_harness_inputs, Mapping)
            else {}
        )
        active_layout_gpt_inputs = active_harness_inputs.get("layout_gpt")
        active_layout_gpt_inputs = (
            active_layout_gpt_inputs
            if isinstance(active_layout_gpt_inputs, Mapping)
            else {}
        )
        readiness = _execution_readiness(
            method,
            configs.get(method),
            offline_artifact=offline,
            allow_offline_artifacts=allow_offline_artifacts,
            catalog=catalog,
            required_api_base_sha256=active_model_policy.get(
                "required_api_base_sha256"
            ),
            required_layoutgpt_icl_sha256=active_layout_gpt_inputs.get(
                "icl_sha256"
            ),
        )
        if offline is None and not runtime_readiness["ready"]:
            readiness = {**readiness, "ready": False,
                         "reasons": list(readiness.get("reasons", [])) + ["evaluator_runtime_not_ready"],
                         "evaluation_runtime": runtime_readiness}
        config = _adapter_config(configs.get(method))
        semantic = check_method_eligibility(
            adapter_name=method,
            protocol=protocol,
            catalog=catalog,
            adapter_config=config,
        )
        model_control = configured_model_policy_report(
            adapter_name=method,
            policy=protocol.as_dict()["generation"].get("model_policy"),
            adapter_config=config,
        )
        if (
            not semantic["eligible"]
            or not model_control["valid"]
            or not readiness["ready"]
        ):
            rows.append(_unattempted_row(
                case_manifest=case, method=method, status="blocked",
                reason="method readiness, eligibility or model policy rejected",
                gate={"execution_readiness": readiness, "semantic_eligibility": semantic,
                      "model_policy": model_control},
            ))
            _write_results(root, rows)
            continue
        if case["case_id"] != cases[0]["case_id"] and method not in passed_dry_run:
            rows.append(_unattempted_row(
                case_manifest=case, method=method, status="skipped",
                reason="first-case full evaluation did not pass; no further generation",
            ))
            _write_results(root, rows)
            continue
        try:
            row = _run_case(
                root=root, case_manifest=case, method=method, catalog=catalog,
                config=configs.get(method), method_output=offline,
                prepared_manifest=manifest,
                evaluation_runtime=evaluation_runtime,
            )
        except (KeyboardInterrupt, SystemExit):
            row = _unattempted_row(
                case_manifest=case, method=method, status="cancelled",
                reason="operator cancellation; cohort stopped",
            )
            row["attempted"] = True
            row.update(_upstream_execution_evidence(root / "cases" / case["case_id"] / method))
            rows.append(row)
            rows.extend(_unattempted_row(
                case_manifest=pending_case, method=pending_method, status="skipped",
                reason="cohort cancelled before this unit",
            ) for pending_case, pending_method in plan[index + 1:])
            _finish_pilot(root, manifest, rows, passed_dry_run, dry_run_only,
                          len(planned_cases), cancelled=True)
            raise
        except Exception as exc:
            # _run_case records execution/evaluation failures itself. An error
            # escaping it occurred at the pre-launch integrity/output gate.
            # Stop the cohort, preserve every planned unit, and propagate it.
            rows.append(_unattempted_row(
                case_manifest=case, method=method, status="blocked",
                reason=f"pre-launch gate failed: {redact_private_locators(str(exc))}",
            ))
            rows.extend(_unattempted_row(
                case_manifest=pending_case, method=pending_method, status="skipped",
                reason="cohort stopped by pre-launch integrity/output gate",
            ) for pending_case, pending_method in plan[index + 1:])
            _finish_pilot(root, manifest, rows, passed_dry_run, dry_run_only, len(planned_cases))
            raise
        rows.append(row)
        if row["valid_comparison_run"] and row["evaluation_success"]:
            if method not in passed_dry_run:
                passed_dry_run.append(method)
        _write_results(root, rows)
    return _finish_pilot(root, manifest, rows, passed_dry_run, dry_run_only,
                         len(planned_cases))


def _finish_pilot(
    root: Path, manifest: Mapping[str, Any], rows: list[dict[str, Any]],
    passed_dry_run: list[str], dry_run_only: bool, planned_case_count: int,
    *, cancelled: bool = False,
) -> dict[str, Any]:
    _write_results(root, rows)
    compatibility = read_json(manifest["compatibility_report"])
    summary = _summarize_results(
        rows,
        methods=list(manifest["methods"]),
        case_count=planned_case_count,
    )
    summary_path = write_json(root / "summary.json", summary)
    updated = dict(manifest)
    attempted = sum(bool(row["attempted"]) for row in rows)
    completed = sum(row["run_status"] == "completed" for row in rows)
    status = (
        "cancelled" if cancelled else "blocked" if attempted == 0
        else "completed" if completed == len(rows)
        else "failed" if attempted == len(rows) and completed == 0 else "partial"
    )
    updated.update(
        {
            "status": status,
            "scheduler_status": "cancelled" if cancelled else "finished",
            "planned_runs": len(rows),
            "experiment_complete": status == "completed",
            "dry_run_only": bool(dry_run_only),
            "dry_run_passed_methods": passed_dry_run,
            "attempted_runs": attempted,
            "unattempted_runs": len(rows) - attempted,
            "valid_runs": sum(bool(row["valid_comparison_run"]) for row in rows),
            "summary": summary_path.resolve().as_posix(),
            "real_upstream_execution_performed": any(
                row.get("upstream_process_started") is True
                for row in rows
            ),
        }
    )
    manifest_path = root / "pilot_manifest.json"
    write_json(manifest_path, updated)
    _write_readme(root, updated, compatibility, summary=summary)
    return {**updated, "manifest_path": manifest_path.resolve().as_posix()}


def _upstream_execution_evidence(method_dir: Path) -> dict[str, Any]:
    """A scheduled attempt is not proof that an upstream process started.

    The common executor only records a subprocess return code after Popen, or
    a timeout after communication started. Callbacks and configured artifacts
    must not be counted as real external process execution. This proves launch,
    not successful native-workflow or production-API certification.
    """
    path = method_dir / "generator" / "execution" / "execution_result.json"
    evidence: dict[str, Any] = {"upstream_process_started": False,
                                "execution_result": None}
    if not path.is_file():
        return evidence
    try:
        value = read_json(path)
    except (ValueError, OSError):
        return evidence
    if not isinstance(value, Mapping):
        return evidence
    evidence.update({
        "execution_result": path.resolve().as_posix(),
        "execution_result_sha256": _file_sha256(path),
        "upstream_process_started": value.get("runner_kind") == "subprocess" and (
            value.get("return_code") is not None or value.get("timed_out") is True
        ),
    })
    return evidence


def _unattempted_row(
    *, case_manifest: Mapping[str, Any], method: str, status: str, reason: str,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _failure_row(
        case_manifest=case_manifest, method=method, execution_mode="not_executed",
        method_dir=Path(case_manifest["case_manifest"]).parent / method,
        error=ArtifactValidationError(reason),
    )
    row.update({"run_status": status, "attempted": False, "generation_success": False,
                "failure_class": status, "failure_source": "orchestration",
                "run_manifest": None,
                "upstream_process_started": False, "execution_result": None,
                "readiness": deepcopy(gate) if gate is not None else None})
    return row


def _run_case(
    *,
    root: Path,
    case_manifest: Mapping[str, Any],
    method: str,
    catalog: CanonicalAssetCatalog,
    config: Mapping[str, Any] | None,
    method_output: str | Path | None,
    prepared_manifest: Mapping[str, Any],
    evaluation_runtime: CanonicalEvaluationRuntime | None = None,
) -> dict[str, Any]:
    case_id = str(case_manifest["case_id"])
    method_dir = root / "cases" / case_id / method
    if method_dir.exists():
        raise FileExistsError(
            f"method/case output already exists and will not be overwritten: {method_dir}"
        )
    # Revalidate immediately before each unit and use the verified parsed bytes,
    # not another unverified read after the gate.
    verified = verify_prepared_artifacts(root, prepared_manifest)
    case = verified["cases"][case_id]
    generation_input = case["generation_input"]
    object_plan = case["object_plan"]
    protocol = case["protocol"]
    evaluator_policy = verified["evaluator_policy"]
    static_evaluator_kwargs = evaluator_policy.get("static_kwargs")
    static_evaluator_kwargs = (
        dict(static_evaluator_kwargs)
        if isinstance(static_evaluator_kwargs, Mapping)
        else {}
    )
    forbidden_evaluator_keys = sorted(
        set(static_evaluator_kwargs) & {"scene", "out", "evaluation_mode"}
    )
    if forbidden_evaluator_keys:
        raise ArtifactValidationError(
            "pilot evaluator static_kwargs cannot control "
            f"{forbidden_evaluator_keys}"
        )
    execution_mode = "offline_artifact" if method_output is not None else "real_generation"
    try:
        result = run_controlled_generation(
            generation_input=generation_input,
            adapter_name=method,
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=method_dir,
            adapter_config=_adapter_config(config),
            method_output=method_output,
            run_generation=method_output is None,
            evaluation_kwargs={
                **static_evaluator_kwargs,
                "scene_request": generation_input["scene_request"],
                "object_plan": object_plan,
            },
            evaluation_runtime=evaluation_runtime,
        )
        evaluation = read_json(result["evaluator"]["report"])
        row = _success_row(
            case_manifest=case_manifest,
            method=method,
            execution_mode=execution_mode,
            result=result,
            evaluation=evaluation,
        )
    except Exception as exc:
        row = _failure_row(
            case_manifest=case_manifest,
            method=method,
            execution_mode=execution_mode,
            method_dir=method_dir,
            error=exc,
        )
        failure_path = method_dir / "pilot_failure.json"
        if failure_path.exists():
            raise FileExistsError(f"refusing to overwrite {failure_path}") from exc
        write_json(
            failure_path,
            {
                "schema_version": "controlled_generation_pilot_failure_v1",
                "case_id": case_id,
                "method": method,
                "failure_class": row["failure_class"],
                "failure_source": row["failure_source"],
                "reason": row["failure_reason"],
            },
        )
    row.update(_upstream_execution_evidence(method_dir))
    return row


def _success_row(
    *,
    case_manifest: Mapping[str, Any],
    method: str,
    execution_mode: str,
    result: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    complexity = case_manifest["complexity"]
    hard = _hard_metric_outcomes(evaluation)
    trajectory = _trajectory_summary(result.get("sceneweaver_trajectory"))
    resources = result.get("generation_resources")
    resources = resources if isinstance(resources, Mapping) else {}
    score = evaluation.get("benchmark_score")
    score_available = (
        isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
    )
    evaluation_success = score_available and evaluation.get("benchmark_score_status") == "complete"
    evaluation_failure_reason = (
        None
        if evaluation_success
        else "canonical evaluator did not produce a complete benchmark score: "
        f"{evaluation.get('benchmark_score_status')}"
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": case_manifest["case_id"],
        "method": method,
        "run_status": "completed" if evaluation_success and result["valid_comparison_run"] else "failed",
        "attempted": True,
        "execution_mode": execution_mode,
        "protocol_id": result["protocol_id"],
        "protocol_hash": result["protocol_sha256"],
        "architecture_hash": result["architecture_sha256"],
        "inventory_hash": result["object_inventory_sha256"],
        "asset_binding_hash": result["asset_binding_sha256"],
        "evaluator_config_hash": case_manifest["evaluator_config_sha256"],
        "seed": case_manifest.get("seed"),
        "seed_enforcement": case_manifest.get("seed_enforcement"),
        **complexity,
        "generation_success": True,
        "valid_comparison_run": bool(result["valid_comparison_run"]),
        "evaluation_success": evaluation_success,
        "score_available": score_available,
        "benchmark_score": score,
        "benchmark_score_100": evaluation.get("benchmark_score_100"),
        "benchmark_score_status": evaluation.get("benchmark_score_status"),
        "layer_scores": {
            name: report.get("score")
            for name, report in (evaluation.get("layer_reports") or {}).items()
            if isinstance(report, Mapping)
        },
        "hard_metric_outcomes": hard,
        "hard_failure_count": sum(
            1 for value in hard.values() if value.get("hard_failure") is True
        ),
        "generation_runtime_seconds": resources.get("wall_clock_seconds"),
        "generation_iterations": resources.get("iteration_count"),
        "tokens": resources.get("tokens"),
        "tool_calls": resources.get("tool_calls"),
        "failure_class": None if evaluation_success else "evaluator_infrastructure_failure",
        "failure_source": None if evaluation_success else "infrastructure",
        "failure_reason": evaluation_failure_reason,
        "run_manifest": result.get("manifest_path"),
        "evaluation_report": result["evaluator"]["report"],
        **trajectory,
    }


def _failure_row(
    *,
    case_manifest: Mapping[str, Any],
    method: str,
    execution_mode: str,
    method_dir: Path,
    error: BaseException,
) -> dict[str, Any]:
    failure_class, failure_source = _classify_failure(method_dir, error)
    complexity = case_manifest["complexity"]
    comparison_manifest = method_dir / "comparison" / "run_manifest.json"
    generation_success = (method_dir / "generated_scene.json").is_file()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": case_manifest["case_id"],
        "method": method,
        "run_status": "failed",
        "attempted": True,
        "execution_mode": execution_mode,
        "protocol_id": "generation_comparison_v1",
        "protocol_hash": case_manifest["protocol_sha256"],
        "architecture_hash": case_manifest["architecture_sha256"],
        "inventory_hash": case_manifest["object_inventory_sha256"],
        "asset_binding_hash": case_manifest["asset_binding_sha256"],
        "evaluator_config_hash": case_manifest["evaluator_config_sha256"],
        "seed": case_manifest.get("seed"),
        "seed_enforcement": case_manifest.get("seed_enforcement"),
        **complexity,
        "generation_success": generation_success,
        "valid_comparison_run": False,
        "evaluation_success": False,
        "score_available": False,
        "benchmark_score": None,
        "benchmark_score_100": None,
        "benchmark_score_status": None,
        "layer_scores": {},
        "hard_metric_outcomes": {},
        "hard_failure_count": None,
        "generation_runtime_seconds": None,
        "generation_iterations": None,
        "tokens": None,
        "tool_calls": None,
        "failure_class": failure_class,
        "failure_source": failure_source,
        "failure_reason": redact_private_locators(str(error)),
        "run_manifest": (
            comparison_manifest.resolve().as_posix()
            if comparison_manifest.is_file()
            else None
        ),
        "evaluation_report": None,
        **_empty_trajectory(),
    }


def _preflight_imaginarium_catalog(
    catalog_spec: Mapping[str, Any],
    asset_root: Path,
    *,
    asset_bundle_root: Path | None = None,
) -> tuple[CanonicalAssetCatalog, dict[str, Any]]:
    csv_path = asset_root / "imaginarium_asset_info.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Imaginarium catalog CSV is missing: {csv_path}")
    csv_sha256 = _file_sha256(csv_path)
    expected_csv_sha256 = catalog_spec.get("source_catalog_csv_sha256")
    if (
        expected_csv_sha256 is not None
        and str(expected_csv_sha256) != csv_sha256
    ):
        raise ArtifactValidationError(
            "Imaginarium catalog CSV differs from the frozen source hash"
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = {str(row.get("name_en")): row for row in csv.DictReader(handle)}
    bundle_validation = None
    if asset_bundle_root is not None:
        bundle_validation = validate_imaginarium_glb_bundle(
            plan=asset_bundle_root / "bundle_plan.json",
            report=asset_bundle_root / "bundle_report.json",
            expected_asset_root=asset_root,
            expected_bundle_root=asset_bundle_root,
        )
        if not bundle_validation["valid"]:
            raise ArtifactValidationError(
                "Imaginarium GLB bundle failed its content/geometry validation"
            )
    records = []
    preflight_rows = []
    for expected in catalog_spec["assets"]:
        asset_id = str(expected["asset_id"])
        row = csv_rows.get(asset_id)
        asset_dir = asset_root / asset_id
        source_mesh_path = asset_dir / f"{asset_id}.fbx"
        mesh_path = source_mesh_path
        if asset_bundle_root is not None:
            try:
                mesh_path = bundle_mesh_path(asset_bundle_root, asset_id)
            except FileNotFoundError:
                mesh_path = asset_bundle_root / asset_id / f"{asset_id}.glb"
        metadata_path = asset_dir / f"{asset_id}_metadata.json"
        errors = []
        if row is None:
            errors.append("catalog_row_missing")
        if not source_mesh_path.is_file() or source_mesh_path.stat().st_size <= 0:
            errors.append("source_mesh_missing_or_empty")
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            errors.append("mesh_missing_or_empty")
        if not metadata_path.is_file():
            errors.append("metadata_missing")
            metadata: dict[str, Any] = {}
        else:
            loaded = read_json(metadata_path)
            metadata = dict(loaded) if isinstance(loaded, Mapping) else {}
        category = str(row.get("category") or "") if row is not None else ""
        description = str(row.get("short_desc") or "") if row is not None else ""
        if category != str(expected["category"]):
            errors.append("category_mismatch")
        if description != str(expected["description"]):
            errors.append("description_mismatch")
        try:
            bbox_size = _vector3(metadata.get("transformed_size"), positive=True)
            bbox_center = _vector3(metadata.get("transformed_bbox_center"))
        except ValueError as exc:
            errors.append(f"bbox_invalid:{exc}")
            bbox_size = [1.0, 1.0, 1.0]
            bbox_center = [0.0, 0.0, 0.0]
        front = expected.get("canonical_front")
        if front is not None:
            try:
                front = _vector3(front)
                if math.sqrt(sum(value * value for value in front)) <= 1.0e-12:
                    raise ValueError("zero canonical front")
            except ValueError as exc:
                errors.append(f"canonical_front_invalid:{exc}")
        source_fbx_hash = (
            _file_sha256(source_mesh_path) if source_mesh_path.is_file() else None
        )
        source_metadata_hash = (
            _file_sha256(metadata_path) if metadata_path.is_file() else None
        )
        if (
            expected.get("source_fbx_sha256") is not None
            and expected.get("source_fbx_sha256") != source_fbx_hash
        ):
            errors.append("source_fbx_hash_mismatch")
        if (
            expected.get("source_metadata_sha256") is not None
            and expected.get("source_metadata_sha256") != source_metadata_hash
        ):
            errors.append("source_metadata_hash_mismatch")
        mesh_hash = _file_sha256(mesh_path) if mesh_path.is_file() else None
        record = {
            "asset_id": asset_id,
            "source_db": str(catalog_spec.get("source_db") or "imaginarium"),
            "category": str(expected["category"]),
            "description": str(expected["description"]),
            "mesh_uri": mesh_path.resolve().as_posix(),
            "bbox_size_local": bbox_size,
            "bbox_center_local": bbox_center,
            "native_scale": [1.0, 1.0, 1.0],
            "content": (
                {
                    "mesh_sha256": mesh_hash,
                    "mesh_bytes": mesh_path.stat().st_size,
                }
                if mesh_hash is not None
                else {}
            ),
            "metadata": {
                "catalog_row_id": row.get("id") if row is not None else None,
                "scaling_strategy": (
                    row.get("scaling_strategy") if row is not None else None
                ),
                "source_metadata_sha256": (
                    source_metadata_hash
                ),
                "source_fbx_sha256": (
                    source_fbx_hash
                ),
                "mesh_materialization": (
                    "verified_external_glb_bundle"
                    if asset_bundle_root is not None
                    else "native_imaginarium_fbx"
                ),
                "canonical_front_source": expected.get(
                    "canonical_front_source"
                ),
            },
        }
        if front is not None:
            record["canonical_front"] = front
        records.append(record)
        preflight_rows.append(
            {
                "asset_id": asset_id,
                "status": "passed" if not errors else "failed",
                "errors": errors,
                "category_expected": expected["category"],
                "category_catalog": category,
                "mesh_path": mesh_path.resolve().as_posix(),
                "source_fbx_path": source_mesh_path.resolve().as_posix(),
                "mesh_exists": mesh_path.is_file(),
                "mesh_sha256": mesh_hash,
                "bbox_size_local": bbox_size,
                "bbox_center_local": bbox_center,
                "native_scale": [1.0, 1.0, 1.0],
                "canonical_front_status": (
                    "validated" if front is not None else "unavailable_not_invented"
                ),
                "canonical_front_source": expected.get(
                    "canonical_front_source"
                ),
            }
        )
    catalog = CanonicalAssetCatalog.from_mapping(
        {
            "catalog_id": catalog_spec["catalog_id"],
            "catalog_version": catalog_spec["catalog_version"],
            "linear_unit": "meter",
            "assets": records,
            "metadata": {
                "backend": "imaginarium_frozen_subset_v1",
                "pilot_asset_count": len(records),
            },
        },
        hash_local_meshes=True,
    )
    failed = [row["asset_id"] for row in preflight_rows if row["status"] != "passed"]
    report = {
        "schema_version": ASSET_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if not failed else "failed",
        "asset_root": asset_root.resolve().as_posix(),
        "source_catalog_csv": csv_path.resolve().as_posix(),
        "source_catalog_csv_sha256": csv_sha256,
        "expected_source_catalog_csv_sha256": expected_csv_sha256,
        "asset_bundle_root": (
            asset_bundle_root.resolve().as_posix()
            if asset_bundle_root is not None
            else None
        ),
        "mesh_format": "glb" if asset_bundle_root is not None else "fbx",
        "asset_bundle_validation": bundle_validation,
        "catalog": catalog.identity,
        "asset_count": len(preflight_rows),
        "passed": len(preflight_rows) - len(failed),
        "failed": len(failed),
        "failed_asset_ids": failed,
        "assets": preflight_rows,
    }
    return catalog, report


def _case_protocol(
    *,
    pilot: Mapping[str, Any],
    case: Mapping[str, Any],
    catalog: CanonicalAssetCatalog,
    evaluator_policy: Mapping[str, Any],
) -> ComparisonProtocol:
    return ComparisonProtocol.from_mapping(
        {
            "protocol_id": pilot["protocol_id"],
            "protocol_version": pilot["protocol_version"],
            "mode": "frozen_assets",
            "case_id": case["case_id"],
            "architecture": {
                "room_model": "single_room",
                "boundary_model": "axis_aligned_rectangle",
                "room": case["room"],
            },
            "object_inventory_policy": "frozen",
            "objects": case["objects"],
            "asset_policy": "frozen_exact",
            "assets": catalog.identity,
            "scale_policy": "fixed_native_scale",
            "retrieval_policy": "disabled_exact_bindings",
            "generation": {
                **dict(pilot.get("generation") or {}),
                "asset_selection_status": pilot.get("asset_selection_status"),
                "seed": case.get("seed"),
                "seed_enforcement": "not_guaranteed_unless_runner_reports",
            },
            "evaluator": dict(evaluator_policy),
        }
    )


def _generation_input_for_case(
    *, pilot: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = {
        "pilot_id": pilot["pilot_id"],
        "case_id": case["case_id"],
        "seed": case.get("seed"),
    }
    if not isinstance(case.get("object_plan"), Mapping):
        return build_direct_natural_language_generation_input(
            request_id=str(case["case_id"]),
            instruction=str(case["instruction"]),
            scene_type=str(case["scene_type"]),
            room=dict(case["room"]),
            metadata=metadata,
        )
    plan = deepcopy(dict(case["object_plan"]))
    plan["request_id"] = str(case["case_id"])
    plan["scene_type"] = str(case["scene_type"])
    plan["scene_description"] = str(case["instruction"])
    validate_object_plan(plan)
    request = build_scene_request(
        request_id=str(case["case_id"]),
        instruction=str(case["instruction"]),
        scene_type=str(case["scene_type"]),
        room=dict(case["room"]),
        structure=True,
        metadata=metadata,
    )
    return build_generation_input(scene_request=request, object_plan=plan)


def _evaluation_object_plan(case: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(case.get("object_plan"), Mapping):
        plan = deepcopy(dict(case["object_plan"]))
        plan["request_id"] = str(case["case_id"])
        plan["scene_type"] = str(case["scene_type"])
        plan["scene_description"] = str(case["instruction"])
        validate_object_plan(plan)
        return plan
    return {
        "request_id": case["case_id"],
        "scene_type": case["scene_type"],
        "scene_description": case["instruction"],
        "objects": [
            {
                "id": item["slot_id"],
                "category": item["category"],
                "description": item["description"],
                "metadata": {"comparison_slot_id": item["slot_id"]},
                "placement_intent": {
                    "absolute_relations": [],
                    "relative_relations": [],
                },
            }
            for item in case["objects"]
        ],
        "global_constraints": [],
        "relations": [],
    }


def _case_complexity(case: Mapping[str, Any]) -> dict[str, Any]:
    boundary = case["room"]["boundary"]
    xs = [float(point[0]) for point in boundary]
    ys = [float(point[1]) for point in boundary]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    count = len(case["objects"])
    return {
        "object_count": count,
        "room_area": float(area),
        "object_density": float(count / area),
        "pairwise_interaction_proxy": count * (count - 1) // 2,
    }


def _compatibility_report(
    *,
    methods: Sequence[str],
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog,
    method_configs: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    generation_policy = protocol.as_dict()["generation"]
    model_policy = generation_policy.get("model_policy")
    harness_inputs = generation_policy.get("harness_inputs")
    harness_inputs = harness_inputs if isinstance(harness_inputs, Mapping) else {}
    layout_gpt_inputs = harness_inputs.get("layout_gpt")
    layout_gpt_inputs = (
        layout_gpt_inputs if isinstance(layout_gpt_inputs, Mapping) else {}
    )
    for method in methods:
        config = _adapter_config(method_configs.get(method))
        semantic = check_method_eligibility(
            adapter_name=method,
            protocol=protocol,
            catalog=catalog,
            adapter_config=config,
        )
        model_control = configured_model_policy_report(
            adapter_name=method,
            policy=model_policy,
            adapter_config=config,
        )
        readiness = _execution_readiness(
            method,
            method_configs.get(method),
            offline_artifact=None,
            allow_offline_artifacts=False,
            catalog=catalog,
            required_api_base_sha256=(
                model_policy.get("required_api_base_sha256")
                if isinstance(model_policy, Mapping)
                else None
            ),
            required_layoutgpt_icl_sha256=(
                str(layout_gpt_inputs["icl_sha256"])
                if layout_gpt_inputs.get("icl_sha256") is not None
                else None
            ),
        )
        if not semantic["eligible"] or not model_control["valid"]:
            status = "INELIGIBLE"
        elif not readiness["ready"]:
            status = "SEMANTICALLY_ELIGIBLE_INFRASTRUCTURE_UNAVAILABLE"
        else:
            status = "ELIGIBLE_READY"
        rows.append(
            {
                "method": method,
                "status": status,
                "semantic_eligibility": semantic,
                "model_policy": model_control,
                "execution_readiness": readiness,
                "config_summary": _public_config_summary(method_configs.get(method)),
            }
        )
    return {
        "schema_version": "controlled_generation_pilot_compatibility_v1",
        "protocol_mode": "frozen_assets",
        "methods": rows,
    }


def _execution_readiness(
    method: str,
    value: Any,
    *,
    offline_artifact: str | Path | None,
    allow_offline_artifacts: bool,
    catalog: CanonicalAssetCatalog | None = None,
    required_api_base_sha256: str | None = None,
    required_layoutgpt_icl_sha256: str | None = None,
) -> dict[str, Any]:
    if offline_artifact is not None:
        path = Path(offline_artifact).expanduser().resolve()
        return {
            "ready": bool(allow_offline_artifacts and path.exists()),
            "mode": "offline_artifact",
            "reasons": [] if allow_offline_artifacts and path.exists() else [
                "offline_artifacts_not_allowed_or_missing"
            ],
        }
    config = _adapter_config(value)
    reasons = []
    if method == "catalog_placement":
        if not str(config.get("endpoint") or "").strip():
            reasons.append("endpoint_missing")
        if not str(config.get("model") or config.get("model_id") or "").strip():
            reasons.append("model_missing")
        key_env = str(config.get("api_key_env") or "").strip()
        if not key_env:
            reasons.append("api_key_env_missing")
        elif not os.environ.get(key_env):
            reasons.append("api_key_environment_unset")
        if config.get("api_key") is not None:
            reasons.append("literal_api_key_forbidden")
    else:
        execution = config.get("execution")
        if not isinstance(execution, Mapping):
            reasons.append("execution_config_missing")
        else:
            repo = execution.get("repo_path")
            if not repo or not Path(str(repo)).expanduser().is_dir():
                reasons.append("upstream_repo_missing")
            else:
                expected_commit = str(
                    execution.get("expected_upstream_commit") or ""
                ).strip()
                if method in {
                    "layout_gpt",
                    "direct_layout",
                    "layout_vlm",
                    "scene_weaver",
                } and (
                    len(expected_commit) != 40
                    or any(
                        character not in "0123456789abcdef"
                        for character in expected_commit.lower()
                    )
                ):
                    reasons.append("expected_upstream_commit_missing_or_invalid")
                elif expected_commit and _git_commit(
                    Path(str(repo)).expanduser().resolve()
                ) != expected_commit:
                    reasons.append("upstream_commit_mismatch")
                if expected_commit and _git_clean(
                    Path(str(repo)).expanduser().resolve()
                ) is not True:
                    reasons.append("upstream_worktree_not_clean")
            if not execution.get("command"):
                reasons.append("execution_command_missing")
            python_value = str(execution.get("python_executable") or "").strip()
            if not python_value:
                reasons.append("python_executable_missing")
            elif "/" in python_value or python_value.startswith("."):
                python_path = Path(python_value).expanduser()
                if not python_path.is_file() or not os.access(python_path, os.X_OK):
                    reasons.append("python_executable_unavailable")
            elif shutil.which(python_value) is None:
                reasons.append("python_executable_unavailable")
            variables = execution.get("template_variables")
            variables = variables if isinstance(variables, Mapping) else {}
            for name in ("bridge_script", "layoutgpt_icl_examples", "frozen_plugin"):
                if name not in variables:
                    continue
                path = Path(str(variables[name])).expanduser()
                if not path.is_file():
                    reasons.append(f"{name}_unavailable")
            expected_entrypoint = str(
                execution.get("expected_entrypoint_sha256") or ""
            ).strip()
            bridge_value = variables.get("bridge_script")
            command_tokens = _readiness_command_tokens(execution.get("command"))
            if bridge_value and not any(
                token in {"{bridge_script}", str(bridge_value)}
                for token in command_tokens
            ):
                reasons.append("bridge_script_not_in_command")
            if method in {
                "layout_gpt",
                "direct_layout",
                "layout_vlm",
                "scene_weaver",
            } and not expected_entrypoint:
                reasons.append("expected_entrypoint_sha256_missing")
            elif expected_entrypoint:
                if len(expected_entrypoint) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in expected_entrypoint
                ):
                    reasons.append("expected_entrypoint_sha256_invalid")
                elif not bridge_value or not Path(str(bridge_value)).expanduser().is_file():
                    reasons.append("expected_entrypoint_unavailable")
                elif _file_sha256(
                    Path(str(bridge_value)).expanduser().resolve()
                ) != expected_entrypoint:
                    reasons.append("entrypoint_sha256_mismatch")
            expected_bundle = str(
                execution.get("expected_bridge_bundle_sha256") or ""
            ).strip()
            if method in {
                "layout_gpt",
                "direct_layout",
                "layout_vlm",
                "scene_weaver",
            } and not expected_bundle:
                reasons.append("expected_bridge_bundle_sha256_missing")
            elif expected_bundle:
                if len(expected_bundle) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in expected_bundle
                ):
                    reasons.append("expected_bridge_bundle_sha256_invalid")
                elif not bridge_value or not Path(str(bridge_value)).expanduser().is_file():
                    reasons.append("expected_bridge_bundle_unavailable")
                elif bridge_bundle_identity(
                    Path(str(bridge_value)).expanduser().resolve()
                )["bridge_bundle_sha256"] != expected_bundle:
                    reasons.append("bridge_bundle_sha256_mismatch")
            if method == "layout_gpt":
                digest = str(variables.get("layoutgpt_icl_sha256") or "")
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    reasons.append("layoutgpt_icl_sha256_invalid")
                else:
                    icl_value = variables.get("layoutgpt_icl_examples")
                    icl_path = (
                        Path(str(icl_value)).expanduser()
                        if icl_value is not None
                        else None
                    )
                    if icl_path is not None and icl_path.is_file() and (
                        _file_sha256(icl_path.resolve()) != digest
                    ):
                        reasons.append("layoutgpt_icl_sha256_mismatch")
                    if (
                        required_layoutgpt_icl_sha256 is not None
                        and digest != required_layoutgpt_icl_sha256
                    ):
                        reasons.append("layoutgpt_icl_protocol_sha256_mismatch")
            if method == "layout_vlm":
                auxiliary = execution.get("auxiliary_artifacts")
                auxiliary = auxiliary if isinstance(auxiliary, Mapping) else {}
                scene_config_artifact = auxiliary.get("scene_config")
                prepared_target = _command_option_value(
                    command_tokens,
                    "--prepared-scene-config-output",
                )
                if not scene_config_artifact:
                    reasons.append("layoutvlm_prepared_scene_config_not_preserved")
                if prepared_target is None:
                    reasons.append("layoutvlm_prepared_scene_config_output_missing")
                elif str(scene_config_artifact) != prepared_target:
                    reasons.append("layoutvlm_prepared_scene_config_path_mismatch")
            if method == "scene_weaver":
                digest = str(variables.get("frozen_plugin_sha256") or "")
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    reasons.append("sceneweaver_plugin_sha256_invalid")
                else:
                    plugin_value = variables.get("frozen_plugin")
                    plugin_path = (
                        Path(str(plugin_value)).expanduser()
                        if plugin_value is not None
                        else None
                    )
                    if plugin_path is not None and plugin_path.is_file() and (
                        _file_sha256(plugin_path.resolve()) != digest
                    ):
                        reasons.append("sceneweaver_plugin_sha256_mismatch")
            environment = execution.get("environment")
            environment = environment if isinstance(environment, Mapping) else {}
            endpoint_name = (
                "LAYOUT_DDD_API_ENDPOINT"
                if method == "layout_gpt"
                else "LAYOUT_DDD_API_BASE_URL"
            )
            endpoint = str(
                environment.get(endpoint_name) or os.environ.get(endpoint_name) or ""
            ).strip()
            if not endpoint or "YOUR-AUTHORIZED-ENDPOINT" in endpoint:
                reasons.append("model_endpoint_unavailable")
            elif required_api_base_sha256 is not None:
                try:
                    observed_endpoint_sha256 = api_base_sha256(
                        endpoint,
                        completion_endpoint=method == "layout_gpt",
                    )
                except ArtifactValidationError:
                    reasons.append("model_endpoint_invalid")
                else:
                    if observed_endpoint_sha256 != required_api_base_sha256:
                        reasons.append("model_endpoint_fingerprint_mismatch")
            if not os.environ.get("LAYOUT_DDD_API_KEY"):
                reasons.append("model_api_key_environment_unset")
            if method in {"direct_layout", "layout_vlm", "scene_weaver"} and catalog:
                non_glb = [
                    asset["asset_id"]
                    for asset in catalog.assets
                    if Path(str(asset.get("mesh_uri") or "")).suffix.lower()
                    != ".glb"
                ]
                if non_glb:
                    reasons.append("verified_glb_bundle_unavailable")
    return {"ready": not reasons, "mode": "real_generation", "reasons": reasons}


def _readiness_command_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(token) for token in value]
    return []


def _command_option_value(tokens: Sequence[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            value = token[len(prefix):]
            return value or None
        if token == option:
            return tokens[index + 1] if index + 1 < len(tokens) else None
    return None


def _adapter_config(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("adapter_config"), Mapping):
        return dict(value["adapter_config"])
    return dict(value) if isinstance(value, Mapping) else {}


def _public_config_summary(value: Any) -> dict[str, Any]:
    config = _adapter_config(value)
    execution = config.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    endpoint = str(config.get("endpoint") or "").strip()
    return {
        "endpoint_configured": bool(endpoint),
        "endpoint_sha256": (
            hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            if endpoint
            else None
        ),
        "model": config.get("model") or config.get("model_id"),
        "api_key_env": config.get("api_key_env"),
        "literal_secret_present": config.get("api_key") is not None,
        "upstream_repo": execution.get("repo_path"),
        "command_configured": bool(execution.get("command")),
        "comparison_support": config.get("comparison_support"),
    }


def _hard_metric_outcomes(report: Mapping[str, Any]) -> dict[str, Any]:
    reports = report.get("reports")
    reports = reports if isinstance(reports, Mapping) else {}
    generic = reports.get("generic_validity")
    generic = generic if isinstance(generic, Mapping) else {}
    metrics = generic.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    result = {}
    for name in ("collision", "support", "oob"):
        item = metrics.get(name)
        item = item if isinstance(item, Mapping) else {}
        score = item.get("score")
        status = item.get("status")
        result[name] = {
            "status": status,
            "score": score,
            "hard_failure": (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and float(score) <= 0.0
            ),
        }
    return result


def _trajectory_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("iterations"), list):
        return _empty_trajectory()
    rows = value["iterations"]
    scores = [
        float(item["benchmark_score"])
        if isinstance(item.get("benchmark_score"), (int, float))
        and not isinstance(item.get("benchmark_score"), bool)
        else None
        for item in rows
    ]
    numeric = [(index, score) for index, score in enumerate(scores) if score is not None]
    initial = numeric[0][1] if numeric else None
    final = numeric[-1][1] if numeric else None
    regressions = sum(
        1
        for left, right in zip(scores, scores[1:])
        if left is not None and right is not None and right < left
    )
    hard_counts = []
    success_at_iteration = None
    for item in rows:
        report_path = item.get("evaluation_report")
        report = read_json(report_path) if report_path else {}
        outcomes = _hard_metric_outcomes(report if isinstance(report, Mapping) else {})
        hard_counts.append(
            sum(1 for metric in outcomes.values() if metric["hard_failure"])
        )
        if (
            success_at_iteration is None
            and isinstance(item.get("benchmark_score"), (int, float))
            and hard_counts[-1] == 0
        ):
            success_at_iteration = item.get("iteration")
    fixes = sum(max(0, left - right) for left, right in zip(hard_counts, hard_counts[1:]))
    hard_regressions = sum(
        max(0, right - left) for left, right in zip(hard_counts, hard_counts[1:])
    )
    return {
        "initial_score": initial,
        "final_score": final,
        "score_delta": (
            final - initial if initial is not None and final is not None else None
        ),
        "iteration_count": len(rows),
        "success_at_iteration": success_at_iteration,
        "trajectory_regression_count": regressions,
        "trajectory_hard_failure_fixes": fixes,
        "trajectory_hard_failure_regressions": hard_regressions,
    }


def _empty_trajectory() -> dict[str, Any]:
    return {
        "initial_score": None,
        "final_score": None,
        "score_delta": None,
        "iteration_count": None,
        "success_at_iteration": None,
        "trajectory_regression_count": None,
        "trajectory_hard_failure_fixes": None,
        "trajectory_hard_failure_regressions": None,
    }


def _classify_failure(method_dir: Path, error: BaseException) -> tuple[str, str]:
    message = str(error).casefold()
    if "timed out" in message or "timeout" in message:
        return "timeout", "infrastructure"
    if "asset" in message and any(
        token in message
        for token in (
            "identity",
            "replacement",
            "binding",
            "asset_id",
            "selected frozen-asset catalog",
        )
    ):
        return "asset_identity_violation", "method"
    if "architecture" in message or "room geometry" in message:
        return "architecture_violation", "method"
    if "fairness validation" in message or "protocol" in message:
        return "protocol_violation", "method"
    if "native" in message or "schema" in message:
        return "invalid_native_artifact", "method"
    if (method_dir / "generated_scene.json").is_file():
        return "evaluator_infrastructure_failure", "infrastructure"
    if any(
        token in message
        for token in (
            "repo is missing",
            "executable is missing",
            "api key",
            "could not reach",
            "http 5",
            "connection",
        )
    ):
        return "generation_failure", "infrastructure"
    if "canonical" in message or "converter" in message:
        return "canonicalization_failure", "benchmark_infrastructure"
    return "generation_failure", "method"


def _write_results(root: Path, rows: list[dict[str, Any]]) -> None:
    jsonl = root / "results.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in RESULT_COLUMNS
                }
            )


def _summarize_results(
    rows: list[dict[str, Any]],
    *,
    methods: list[str],
    case_count: int,
) -> dict[str, Any]:
    method_rows = {}
    for method in methods:
        planned = [row for row in rows if row["method"] == method]
        selected = [row for row in planned if row.get("attempted", True)]
        valid = [row for row in selected if row["valid_comparison_run"]]
        complete = [row for row in valid if row.get("evaluation_success")]
        scores = [
            float(row["benchmark_score"])
            for row in complete
            if isinstance(row.get("benchmark_score"), (int, float))
            and not isinstance(row.get("benchmark_score"), bool)
            and math.isfinite(row["benchmark_score"])
        ]
        runtimes = [
            float(row["generation_runtime_seconds"])
            for row in selected
            if isinstance(row.get("generation_runtime_seconds"), (int, float))
        ]
        method_rows[method] = {
            "attempted_cases": len(selected),
            "planned_cases": case_count,
            "unattempted_cases": len(planned) - len(selected),
            "unattempted": [
                {"case_id": row["case_id"], "status": row["run_status"],
                 "reason": row["failure_reason"], "readiness": row.get("readiness")}
                for row in planned if not row.get("attempted", True)
            ],
            "valid_runs": len(valid),
            "complete_evaluations": len(complete),
            "incomplete_evaluations": len(valid) - len(complete),
            "scored_runs": len(scores),
            "generation_success_rate": (
                sum(bool(row["generation_success"]) for row in selected) / len(selected)
                if selected
                else None
            ),
            "mean_score": statistics.fmean(scores) if scores else None,
            "median_score": statistics.median(scores) if scores else None,
            "per_case_scores": {
                row["case_id"]: row["benchmark_score"] for row in selected
            },
            "hard_failure_rate": (
                sum((row.get("hard_failure_count") or 0) > 0 for row in valid)
                / len(valid)
                if valid
                else None
            ),
            "average_runtime_seconds": (
                statistics.fmean(runtimes) if runtimes else None
            ),
            "failures": [
                {
                    "case_id": row["case_id"],
                    "class": row["failure_class"],
                    "source": row["failure_source"],
                    "reason": row["failure_reason"],
                }
                for row in selected
                if row["failure_class"] is not None
            ],
        }
    valid_by_case: dict[str, dict[str, float]] = {}
    for row in rows:
        score = row.get("benchmark_score")
        if (row["valid_comparison_run"] and row.get("evaluation_success")
                and isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score)):
            valid_by_case.setdefault(row["case_id"], {})[row["method"]] = float(score)
    paired = []
    for case_id, values in sorted(valid_by_case.items()):
        names = sorted(values)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                paired.append(
                    {
                        "case_id": case_id,
                        "method_a": left,
                        "method_b": right,
                        "score_a_minus_b": values[left] - values[right],
                    }
                )
    return {
        "schema_version": "controlled_generation_pilot_summary_v1",
        "label": "pilot / integration validation",
        "score_aggregation_policy": "valid_comparison_and_complete_evaluation_only",
        "attempted_runs": sum(bool(row.get("attempted", True)) for row in rows),
        "valid_runs": sum(bool(row["valid_comparison_run"]) for row in rows),
        "methods": method_rows,
        "paired_score_deltas": paired,
        "statistical_significance_tested": False,
    }


def _write_readme(
    root: Path,
    manifest: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    *,
    summary: Mapping[str, Any] | None,
) -> None:
    statuses = {
        row["method"]: row["status"] for row in compatibility.get("methods", [])
    }
    lines = [
        "# Controlled FrozenAssets Pilot",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "This is a small pilot / integration validation, not a statistically powered benchmark.",
        "",
        "## Method readiness",
        "",
    ]
    lines.extend(f"- `{method}`: `{status}`" for method, status in statuses.items())
    lines.extend(
        [
            "",
            "Official evaluator results are post-hoc and are never sent to generators.",
            "Failed or partial method/case directories are never overwritten.",
        ]
    )
    if summary is not None:
        lines.extend(
            [
                "",
                "## Result count",
                "",
                f"- Attempted: {summary['attempted_runs']}",
                f"- Valid: {summary['valid_runs']}",
            ]
        )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_pilot_spec(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"pilot schema_version must be {PILOT_SCHEMA_VERSION}"
        )
    if value.get("mode") != "frozen_assets":
        raise ArtifactValidationError("first controlled pilot must use frozen_assets")
    asset_selection_status = value.get("asset_selection_status")
    if asset_selection_status is not None and asset_selection_status not in {
        ASSET_SELECTION_PENDING,
        ASSET_SELECTION_APPROVED,
    }:
        raise ArtifactValidationError(
            "pilot asset_selection_status must be "
            f"{ASSET_SELECTION_PENDING!r} or {ASSET_SELECTION_APPROVED!r}"
        )
    for field in ("pilot_id", "protocol_id"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ArtifactValidationError(f"pilot {field} is required")
    if not isinstance(value.get("catalog"), Mapping):
        raise ArtifactValidationError("pilot catalog is required")
    evaluator = value.get("evaluator")
    if not isinstance(evaluator, Mapping) or not isinstance(
        evaluator.get("static_kwargs", {}), Mapping
    ):
        raise ArtifactValidationError(
            "pilot evaluator and evaluator.static_kwargs must be objects"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not 5 <= len(cases) <= 10:
        raise ArtifactValidationError("pilot must contain 5-10 cases")
    case_ids = [str(case.get("case_id") or "") for case in cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ArtifactValidationError("pilot case_id values must be non-empty and unique")
    methods = value.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ArtifactValidationError("pilot methods must be a non-empty list")
    if len([str(method) for method in methods]) != len(
        set(str(method) for method in methods)
    ):
        raise ArtifactValidationError("pilot methods must not contain duplicates")
    catalog_assets = value["catalog"].get("assets")
    if not isinstance(catalog_assets, list) or not catalog_assets:
        raise ArtifactValidationError("pilot catalog assets must be a non-empty list")
    catalog_by_id = {str(item.get("asset_id")): item for item in catalog_assets}
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("objects"), list):
            raise ArtifactValidationError("every pilot case requires objects")
        object_plan = case.get("object_plan")
        if object_plan is not None:
            if not isinstance(object_plan, Mapping):
                raise ArtifactValidationError("pilot case object_plan must be an object")
            validate_object_plan(dict(object_plan))
            frozen_slots = {str(slot.get("slot_id") or "") for slot in case["objects"]}
            plan_slots = {
                str(item.get("id") or "")
                for item in object_plan.get("objects", [])
                if isinstance(item, Mapping)
            }
            if frozen_slots != plan_slots:
                raise ArtifactValidationError(
                    f"case {case['case_id']} object_plan IDs must exactly match frozen "
                    f"slots; missing={sorted(frozen_slots - plan_slots)}, "
                    f"unexpected={sorted(plan_slots - frozen_slots)}"
                )
            non_unit_counts = {
                str(item.get("id")): item.get("count")
                for item in object_plan.get("objects", [])
                if isinstance(item, Mapping) and int(item.get("count", 1)) != 1
            }
            if non_unit_counts:
                raise ArtifactValidationError(
                    f"case {case['case_id']} object_plan must expand every instance "
                    f"into a unique slot; counts={non_unit_counts}"
                )
        if case.get("source_provenance") is not None and not isinstance(
            case.get("source_provenance"), Mapping
        ):
            raise ArtifactValidationError("pilot case source_provenance must be an object")
        for slot in case["objects"]:
            asset_id = str(slot.get("asset_id") or "")
            asset = catalog_by_id.get(asset_id)
            if asset is None:
                raise ArtifactValidationError(
                    f"case {case['case_id']} references unknown asset {asset_id!r}"
                )
            if slot.get("category") != asset.get("category"):
                raise ArtifactValidationError(
                    f"case {case['case_id']} slot category conflicts with asset {asset_id}"
                )


def _method_configs(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if value is None:
        return {}
    loaded = _load_mapping(value, "method configs")
    methods = loaded.get("methods") if isinstance(loaded.get("methods"), Mapping) else loaded
    return {str(key): item for key, item in methods.items()}


def _load_mapping(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        loaded = read_json(value)
    else:
        loaded = value
    if not isinstance(loaded, Mapping):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    return dict(loaded)


def _vector3(value: Any, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("expected finite 3-vector")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("expected finite 3-vector")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError("expected positive 3-vector")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", root.as_posix(), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def bridge_execution_hashes(entrypoint: str | Path) -> dict[str, Any]:
    """Return the exact source pins expected by publication runner configs.

    The bundle digest is intentionally not a plain file SHA-256: it commits to
    the bridge entrypoint and its sibling ``_common.py`` when present.  Keeping
    the calculation here gives operators one canonical command for replacing
    the fail-closed placeholders in the example configuration.
    """

    path = Path(entrypoint).expanduser().resolve()
    if not path.is_file():
        raise ArtifactValidationError(f"bridge entrypoint is missing: {path}")
    bundle = bridge_bundle_identity(path)
    return {
        "entrypoint": path.as_posix(),
        **bundle,
        "expected_entrypoint_sha256": _file_sha256(path),
        "expected_bridge_bundle_sha256": bundle["bridge_bundle_sha256"],
    }


def _git_clean(root: Path) -> bool | None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            root.as_posix(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return not completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--asset-root", required=True)
    prepare.add_argument("--asset-bundle-root")
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--method-configs")
    prepare.add_argument("--evaluation-runtime-config", help="trusted Judge/Blender runtime JSON; required before real generation")
    prepare.add_argument(
        "--case-id", action="append", dest="case_ids",
        help="select an existing case; repeat for a shared subset (source order, fresh output)",
    )
    run = subparsers.add_parser("run")
    run.add_argument("--prepared-dir", required=True)
    run.add_argument("--method-configs", required=True)
    run.add_argument("--dry-run-only", action="store_true")
    preflight = subparsers.add_parser("preflight", help="read-only checks; NEVER generates, renders or calls a model")
    preflight.add_argument("--prepared-dir", required=True)
    preflight.add_argument("--method-configs")
    source_hash = subparsers.add_parser(
        "hash-bridge",
        help="print the entrypoint and canonical bridge-bundle source pins",
    )
    source_hash.add_argument("--entrypoint", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_controlled_pilot(
            spec=args.spec,
            asset_root=args.asset_root,
            out_dir=args.out_dir,
            method_configs=args.method_configs,
            asset_bundle_root=args.asset_bundle_root,
            evaluation_runtime_config=args.evaluation_runtime_config,
            case_ids=args.case_ids,
        )
    elif args.command == "preflight":
        result = preflight_prepared_pilot(prepared_dir=args.prepared_dir,
                                          method_configs=args.method_configs)
    elif args.command == "run":
        result = run_prepared_pilot(
            prepared_dir=args.prepared_dir,
            method_configs=args.method_configs,
            dry_run_only=args.dry_run_only,
        )
    else:
        result = bridge_execution_hashes(args.entrypoint)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "run" and result["status"] != "completed":
        raise SystemExit(2)
    if args.command == "preflight" and not result["ready_for_generation"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "ASSET_SELECTION_APPROVED",
    "ASSET_SELECTION_PENDING",
    "ASSET_PREFLIGHT_SCHEMA_VERSION",
    "FAILURE_CLASSES",
    "PILOT_MANIFEST_SCHEMA_VERSION",
    "PILOT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "bridge_execution_hashes",
    "prepare_controlled_pilot",
    "preflight_prepared_pilot",
    "run_prepared_pilot",
]
