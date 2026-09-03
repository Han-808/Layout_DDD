"""Controlled generation orchestration on top of existing adapters/evaluator."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.adapters.common.execution import artifact_sha256
from benchmark.api.evaluation import run_evaluate
from benchmark.api.generation import run_generate
from benchmark.api.scene_weaver_iterations import evaluate_scene_weaver_iterations
from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    load_asset_catalog,
)
from benchmark.generation_comparison.eligibility import check_method_eligibility
from benchmark.generation_comparison.identity import (
    architecture_from_generation_input,
    architecture_sha256,
)
from benchmark.generation_comparison.inputs import build_controlled_generation_input
from benchmark.generation_comparison.materializers import (
    architecture_from_native_input,
    materialize_method_catalog,
)
from benchmark.generation_comparison.native_identity import (
    inspect_native_asset_selections,
)
from benchmark.generation_comparison.protocol import (
    FROZEN_ASSETS,
    NATIVE,
    ComparisonProtocol,
    load_comparison_protocol,
)
from benchmark.generation_comparison.validation import validate_comparison_run
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import read_json, write_json


COMPARISON_RUN_MANIFEST_SCHEMA_VERSION = "generation_comparison_run_manifest_v1"


class ComparisonRunError(ArtifactValidationError):
    """Raised after a fail-closed comparison manifest has been persisted."""


def run_controlled_generation(
    *,
    generation_input: Mapping[str, Any],
    adapter_name: str,
    protocol: ComparisonProtocol | Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    asset_catalog: CanonicalAssetCatalog | Mapping[str, Any] | str | Path | None = None,
    adapter_config: Mapping[str, Any] | None = None,
    method_output: str | Path | None = None,
    run_generation: bool = True,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    evaluate_sceneweaver_trajectory: bool = True,
) -> dict[str, Any]:
    """Run one auditable comparison case without altering conversion/evaluation."""

    source_input = deepcopy(dict(generation_input))
    validate_generation_input(source_input)
    contract = load_comparison_protocol(protocol)
    catalog = (
        load_asset_catalog(asset_catalog, hash_local_meshes=True)
        if asset_catalog is not None
        else None
    )
    if contract.mode != NATIVE and catalog is None:
        raise ArtifactValidationError(
            f"{contract.mode} comparison requires asset_catalog"
        )
    if contract.mode == NATIVE and catalog is not None:
        raise ArtifactValidationError(
            "native comparison must use the method's native asset source; "
            "omit asset_catalog"
        )
    root = Path(out_dir)
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = write_json(comparison_dir / "protocol.json", contract.as_dict())
    protocol_file_hash_before = _file_sha256(protocol_path)

    config = _copy_adapter_config(adapter_config)
    materialization: dict[str, Any] | None = None
    if catalog is not None:
        materialization = materialize_method_catalog(
            adapter_name=adapter_name,
            catalog=catalog,
            protocol=contract,
            out_dir=comparison_dir / "method_input" / adapter_name,
        )
    eligibility = check_method_eligibility(
        adapter_name=adapter_name,
        protocol=contract,
        catalog=catalog,
        adapter_config=config,
    )
    input_architecture = architecture_sha256(
        architecture_from_generation_input(source_input)
    )
    if input_architecture != contract.architecture_hash:
        eligibility = _append_ineligibility(
            eligibility,
            code="architecture_mismatch",
            message="public generation input architecture differs from comparison case",
            expected=contract.architecture_hash,
            actual=input_architecture,
        )
    active_architecture = _active_architecture_features(source_input)
    if active_architecture:
        eligibility = _append_ineligibility(
            eligibility,
            code="unsupported_protocol_semantics",
            message=(
                "generation comparison v1 does not allow active walls/openings/"
                "topology in the common track"
            ),
            active_architecture=active_architecture,
        )
    eligibility_path = write_json(comparison_dir / "eligibility.json", eligibility)
    if not eligibility["eligible"]:
        manifest = _base_manifest(
            adapter_name=adapter_name,
            contract=contract,
            catalog=catalog,
            protocol_path=protocol_path,
            eligibility=eligibility,
            eligibility_path=eligibility_path,
            materialization=materialization,
        )
        manifest.update({"status": "INELIGIBLE", "valid_comparison_run": False})
        manifest_path = write_json(comparison_dir / "run_manifest.json", manifest)
        raise ComparisonRunError(
            "comparison pre-run eligibility failed; "
            f"manifest={manifest_path.resolve().as_posix()}"
        )

    controlled_input = build_controlled_generation_input(
        source_input,
        protocol=contract,
        catalog=catalog,
        materialization=materialization,
    )
    controlled_input_path = write_json(
        comparison_dir / "controlled_generation_input.json",
        controlled_input,
    )
    config = _configure_adapter(
        adapter_name=adapter_name,
        config=config,
        contract=contract,
        materialization=materialization,
    )
    result: dict[str, Any] | None = None
    generation_started = time.monotonic()
    try:
        result = run_generate(
            generation_input=controlled_input,
            adapter_name=adapter_name,
            out_dir=root,
            method_output=method_output,
            adapter_config=config,
            run_generation=run_generation,
        )
        if not result.get("generated_scene"):
            raise ComparisonRunError(
                "controlled comparison requires a generated canonical scene"
            )
    except BaseException as exc:
        manifest = _base_manifest(
            adapter_name=adapter_name,
            contract=contract,
            catalog=catalog,
            protocol_path=protocol_path,
            eligibility=eligibility,
            eligibility_path=eligibility_path,
            materialization=materialization,
        )
        manifest.update(
            {
                "status": "GENERATION_FAILED",
                "valid_comparison_run": False,
                "controlled_generation_input": controlled_input_path.resolve().as_posix(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(comparison_dir / "run_manifest.json", manifest)
        raise

    adapter_metadata = read_json(result["adapter_metadata"])
    execution_metadata = adapter_metadata.get("generation_run")
    if not isinstance(execution_metadata, Mapping):
        raise ComparisonRunError("adapter metadata lacks generation_run audit record")
    execution_metadata = dict(execution_metadata)
    execution_metadata.setdefault(
        "runtime_seconds", time.monotonic() - generation_started
    )
    native_artifact = result.get("raw_native_artifact") or result.get("native_output")
    if not native_artifact:
        raise ComparisonRunError("controlled run lacks a preserved native artifact")
    native_selection = inspect_native_asset_selections(
        adapter_name=adapter_name,
        native_artifact=native_artifact,
        execution_metadata=execution_metadata,
        adapter_config=config,
    )
    native_selection_path = write_json(
        comparison_dir / "native_asset_selection.json",
        native_selection,
    )
    method_input = read_json(result["method_input"])
    execution_input = method_input.get("execution_input")
    execution_input = execution_input if isinstance(execution_input, Mapping) else {}
    native_input_path = execution_input.get("path")
    if not native_input_path and adapter_name == "catalog_placement":
        native_input_path = result["method_input"]
    if not native_input_path:
        raise ComparisonRunError("controlled run lacks its prepared native input")
    native_input = read_json(native_input_path)
    method_architecture_hash = architecture_sha256(
        architecture_from_native_input(adapter_name, native_input)
    )
    scene = read_json(result["generated_scene"])
    selected_iteration = _selected_iteration(scene)
    validation = validate_comparison_run(
        adapter_name=adapter_name,
        protocol=contract,
        catalog=catalog,
        canonical_scene=scene,
        materialization=materialization,
        native_selection=native_selection,
        method_input_architecture_sha256=method_architecture_hash,
        eligibility=eligibility,
        selected_iteration=selected_iteration,
    )
    _append_input_immutability_violations(
        validation,
        protocol_path=protocol_path,
        protocol_hash_before=protocol_file_hash_before,
        catalog=catalog,
        materialization=materialization,
    )
    validation_path = write_json(comparison_dir / "validation.json", validation)
    if not validation["valid_comparison_run"]:
        manifest = _completed_manifest(
            adapter_name=adapter_name,
            contract=contract,
            catalog=catalog,
            protocol_path=protocol_path,
            eligibility=eligibility,
            eligibility_path=eligibility_path,
            materialization=materialization,
            result=result,
            execution_metadata=execution_metadata,
            controlled_input_path=controlled_input_path,
            native_selection_path=native_selection_path,
            validation=validation,
            validation_path=validation_path,
            evaluation_report_path=None,
            evaluation_report=None,
            trajectory=None,
        )
        manifest["status"] = "INVALID_COMPARISON"
        manifest_path = write_json(comparison_dir / "run_manifest.json", manifest)
        raise ComparisonRunError(
            "comparison fairness validation failed; "
            f"manifest={manifest_path.resolve().as_posix()}"
        )

    evaluation_options = dict(evaluation_kwargs or {})
    forbidden = sorted(
        key
        for key in ("scene", "out", "evaluation_mode")
        if key in evaluation_options
    )
    if forbidden:
        raise ArtifactValidationError(
            f"controlled comparison owns evaluator arguments {forbidden}"
        )
    evaluation_report_path = root / "evaluation_report.json"
    evaluation_report = run_evaluate(
        scene=scene,
        out=evaluation_report_path,
        **evaluation_options,
    )

    trajectory: dict[str, Any] | None = None
    if adapter_name == "scene_weaver" and evaluate_sceneweaver_trajectory:
        trajectory = _evaluate_sceneweaver_comparison_trajectory(
            native_artifact=Path(native_artifact),
            controlled_input=controlled_input,
            out_dir=comparison_dir / "sceneweaver_trajectory",
            adapter_config=config,
            evaluation_kwargs=evaluation_options,
            protocol=contract,
            catalog=catalog,
            materialization=materialization,
            native_selection=native_selection,
            method_architecture_hash=method_architecture_hash,
            eligibility=eligibility,
        )
        if not trajectory["valid_comparison_trajectory"]:
            validation["valid_comparison_run"] = False
            validation["violations"].append(
                {
                    "code": "sceneweaver_trajectory_violation",
                    "message": "one or more SceneWeaver iterations violate the protocol",
                    "details": {
                        "invalid_iterations": trajectory["invalid_iterations"]
                    },
                }
            )
            write_json(validation_path, validation)

    manifest = _completed_manifest(
        adapter_name=adapter_name,
        contract=contract,
        catalog=catalog,
        protocol_path=protocol_path,
        eligibility=eligibility,
        eligibility_path=eligibility_path,
        materialization=materialization,
        result=result,
        execution_metadata=execution_metadata,
        controlled_input_path=controlled_input_path,
        native_selection_path=native_selection_path,
        validation=validation,
        validation_path=validation_path,
        evaluation_report_path=evaluation_report_path,
        evaluation_report=evaluation_report,
        trajectory=trajectory,
    )
    if not validation["valid_comparison_run"]:
        manifest["status"] = "INVALID_COMPARISON"
    manifest_path = write_json(comparison_dir / "run_manifest.json", manifest)
    if not validation["valid_comparison_run"]:
        raise ComparisonRunError(
            "SceneWeaver trajectory violates the controlled protocol; "
            f"manifest={manifest_path.resolve().as_posix()}"
        )
    return {**manifest, "manifest_path": manifest_path.resolve().as_posix()}


def _configure_adapter(
    *,
    adapter_name: str,
    config: dict[str, Any],
    contract: ComparisonProtocol,
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    configured = dict(config)
    configured["asset_resolution_policy"] = "exact_only"
    if materialization is None:
        return configured
    configured["asset_manifest_path"] = materialization[
        "converter_asset_manifest_path"
    ]
    configured["comparison_control_path"] = materialization[
        "comparison_control_path"
    ]
    execution = configured.get("execution")
    if isinstance(execution, Mapping):
        execution_copy = dict(execution)
        variables = dict(execution_copy.get("template_variables") or {})
        variables.update(
            {
                "comparison_input": materialization["comparison_control_path"],
                "comparison_catalog": materialization["method_catalog_path"],
            }
        )
        if materialization.get("method_asset_root"):
            variables["comparison_asset_root"] = materialization[
                "method_asset_root"
            ]
        environment = dict(execution_copy.get("environment") or {})
        environment.update(
            {
                "LAYOUT_DDD_COMPARISON_INPUT": materialization[
                    "comparison_control_path"
                ],
                "LAYOUT_DDD_COMPARISON_CATALOG": materialization[
                    "method_catalog_path"
                ],
            }
        )
        if materialization.get("method_asset_root"):
            environment["LAYOUT_DDD_COMPARISON_ASSET_ROOT"] = materialization[
                "method_asset_root"
            ]
        execution_copy["template_variables"] = variables
        execution_copy["environment"] = environment
        configured["execution"] = execution_copy
    if adapter_name == "layout_vlm" and contract.mode == FROZEN_ASSETS:
        payload = read_json(materialization["method_catalog_path"])
        scene_config = _configured_layout_vlm_scene(configured)
        scene_config["assets"] = payload["frozen_assets"]
        configured.pop("layout_vlm_scene_config_path", None)
        configured.pop("scene_config_path", None)
        configured.pop("scene_config", None)
        configured["layout_vlm_scene_config"] = scene_config
    return configured


def _configured_layout_vlm_scene(config: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("layout_vlm_scene_config", "scene_config"):
        value = config.get(key)
        if isinstance(value, Mapping):
            return deepcopy(dict(value))
    for key in ("layout_vlm_scene_config_path", "scene_config_path"):
        value = config.get(key)
        if value:
            path = Path(str(value)).expanduser()
            execution = config.get("execution")
            execution = execution if isinstance(execution, Mapping) else {}
            if not path.is_absolute() and execution.get("repo_path"):
                path = Path(str(execution["repo_path"])).expanduser() / path
            loaded = read_json(path)
            if not isinstance(loaded, Mapping):
                raise ArtifactValidationError(
                    "LayoutVLM configured scene input must be a JSON object"
                )
            return deepcopy(dict(loaded))
    return {}


def _evaluate_sceneweaver_comparison_trajectory(
    *,
    native_artifact: Path,
    controlled_input: dict[str, Any],
    out_dir: Path,
    adapter_config: Mapping[str, Any],
    evaluation_kwargs: Mapping[str, Any],
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    materialization: Mapping[str, Any] | None,
    native_selection: Mapping[str, Any],
    method_architecture_hash: str,
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    summary = evaluate_scene_weaver_iterations(
        native_output=native_artifact,
        generation_input=controlled_input,
        out_dir=out_dir,
        adapter_config=adapter_config,
        evaluation_kwargs=evaluation_kwargs,
    )
    rows = []
    invalid = []
    for row in summary["iterations"]:
        iteration = int(row["iteration"])
        scene = read_json(row["canonical_scene"])
        validation = validate_comparison_run(
            adapter_name="scene_weaver",
            protocol=protocol,
            catalog=catalog,
            canonical_scene=scene,
            materialization=materialization,
            native_selection=native_selection,
            method_input_architecture_sha256=method_architecture_hash,
            eligibility=eligibility,
            selected_iteration=iteration,
        )
        path = write_json(
            out_dir / "iterations" / f"iteration_{iteration:03d}" / "comparison_validation.json",
            validation,
        )
        rows.append(
            {
                "iteration": iteration,
                "validation": path.resolve().as_posix(),
                "valid_comparison_run": validation["valid_comparison_run"],
                "selected_asset_ids": validation["selected_asset_ids"],
                "native_artifact": row["native_artifact"],
                "native_artifact_sha256": row["native_artifact_sha256"],
                "canonical_scene": row["canonical_scene"],
                "evaluation_report": row["evaluation_report"],
                "evaluation_workflow": row["evaluation_workflow"],
                "benchmark_score": row.get("benchmark_score"),
            }
        )
        if not validation["valid_comparison_run"]:
            invalid.append(iteration)
    result = {
        "schema_version": "sceneweaver_comparison_trajectory_v1",
        "native_iteration_summary": summary["summary_path"],
        "benchmark_feedback_used_by_native_loop": False,
        "valid_comparison_trajectory": not invalid,
        "invalid_iterations": invalid,
        "iterations": rows,
    }
    path = write_json(out_dir / "comparison_trajectory.json", result)
    return {**result, "summary_path": path.resolve().as_posix()}


def _completed_manifest(
    *,
    adapter_name: str,
    contract: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    protocol_path: Path,
    eligibility: Mapping[str, Any],
    eligibility_path: Path,
    materialization: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    execution_metadata: Mapping[str, Any],
    controlled_input_path: Path,
    native_selection_path: Path,
    validation: Mapping[str, Any],
    validation_path: Path,
    evaluation_report_path: Path | None,
    evaluation_report: Mapping[str, Any] | None,
    trajectory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = _base_manifest(
        adapter_name=adapter_name,
        contract=contract,
        catalog=catalog,
        protocol_path=protocol_path,
        eligibility=eligibility,
        eligibility_path=eligibility_path,
        materialization=materialization,
    )
    canonical_path = Path(str(result["generated_scene"]))
    canonical_hash, _ = artifact_sha256(canonical_path)
    native_hash = execution_metadata.get("native_artifact_sha256")
    if native_hash is None:
        native_hash, _ = artifact_sha256(Path(str(result["raw_native_artifact"])))
    manifest.update(
        {
            "status": "COMPLETED",
            "valid_comparison_run": validation["valid_comparison_run"],
            "controlled_generation_input": controlled_input_path.resolve().as_posix(),
            "method_input": str(result["method_input"]),
            "adapter_metadata": str(result["adapter_metadata"]),
            "execution_result": execution_metadata.get("execution_result_path"),
            "runner": {
                "kind": execution_metadata.get("runner_kind")
                or execution_metadata.get("provider"),
                "source_provenance": execution_metadata.get("runner_provenance"),
                "command": execution_metadata.get("command"),
                "return_code": execution_metadata.get("return_code"),
                "timed_out": execution_metadata.get("timed_out"),
                "stdout": execution_metadata.get("stdout_path"),
                "stderr": execution_metadata.get("stderr_path"),
                "raw_model_response": execution_metadata.get("raw_response_path"),
                "raw_model_response_sha256": execution_metadata.get(
                    "raw_response_sha256"
                ),
                "request_metadata": execution_metadata.get(
                    "request_metadata_path"
                ),
            },
            "native_artifact": str(result["raw_native_artifact"]),
            "native_artifact_sha256": native_hash,
            "native_selection": native_selection_path.resolve().as_posix(),
            "canonical_scene": canonical_path.resolve().as_posix(),
            "canonical_scene_sha256": canonical_hash,
            "validation": validation_path.resolve().as_posix(),
            "architecture_hashes": validation["architecture"],
            "observed_object_inventory_sha256": validation[
                "observed_object_inventory_sha256"
            ],
            "selected_asset_ids": validation["selected_asset_ids"],
            "observed_asset_binding_sha256": validation["asset_binding_sha256"],
            "retrieval_selection_provenance": _retrieval_selection_provenance(
                execution_metadata
            ),
            "upstream": {
                "repo": execution_metadata.get("upstream_repo"),
                "commit": execution_metadata.get("upstream_commit"),
            },
            "generation_resources": _resource_metadata(execution_metadata, contract),
            "evaluator": _evaluator_metadata(
                evaluation_report_path, evaluation_report, contract
            ),
            "sceneweaver_trajectory": trajectory,
        }
    )
    return manifest


def _base_manifest(
    *,
    adapter_name: str,
    contract: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    protocol_path: Path,
    eligibility: Mapping[str, Any],
    eligibility_path: Path,
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = contract.as_dict()
    return {
        "schema_version": COMPARISON_RUN_MANIFEST_SCHEMA_VERSION,
        "method": adapter_name,
        "adapter": adapter_name,
        "protocol_id": payload["protocol_id"],
        "protocol_version": payload["protocol_version"],
        "protocol_mode": payload["mode"],
        "protocol_sha256": contract.sha256,
        "protocol_path": protocol_path.resolve().as_posix(),
        "case_id": payload["case_id"],
        "architecture_sha256": payload["architecture_sha256"],
        "catalog_id": catalog.catalog_id if catalog is not None else None,
        "catalog_version": catalog.catalog_version if catalog is not None else None,
        "catalog_sha256": catalog.sha256 if catalog is not None else None,
        "object_inventory_sha256": payload["object_inventory_sha256"],
        "asset_binding_sha256": payload["asset_binding_sha256"],
        "scale_policy": payload["scale_policy"],
        "retrieval_policy": payload["retrieval_policy"],
        "eligibility": eligibility_path.resolve().as_posix(),
        "eligibility_status": eligibility["status"],
        "control_evidence": deepcopy(eligibility.get("control_evidence")),
        "materialization": dict(materialization) if materialization is not None else None,
    }


def _resource_metadata(
    execution: Mapping[str, Any],
    contract: ComparisonProtocol,
) -> dict[str, Any]:
    callback = execution.get("callback_metadata")
    callback = callback if isinstance(callback, Mapping) else {}
    declared = callback.get("resource_usage")
    declared = dict(declared) if isinstance(declared, Mapping) else {}
    request_metadata: dict[str, Any] = {}
    request_metadata_path = execution.get("request_metadata_path")
    if request_metadata_path:
        loaded = read_json(request_metadata_path)
        request_metadata = dict(loaded) if isinstance(loaded, Mapping) else {}
    usage = request_metadata.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return {
        "budget_policy": contract.as_dict()["generation"]["budget_policy"],
        "wall_clock_seconds": execution.get("runtime_seconds"),
        "model": declared.get("model") or execution.get("model"),
        "generation_calls": declared.get("generation_calls")
        or (1 if request_metadata else None),
        "tokens": declared.get("tokens") or usage.get("total_tokens"),
        "iteration_count": declared.get("iteration_count")
        or (
            len(execution.get("sceneweaver_available_iterations") or [])
            if execution.get("sceneweaver_available_iterations") is not None
            else None
        ),
        "tool_calls": declared.get("tool_calls"),
        "retrieval_calls": declared.get("retrieval_calls"),
        "rendering_calls": declared.get("rendering_calls"),
        "reported_by_upstream": declared,
    }


def _retrieval_selection_provenance(
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    callback = execution.get("callback_metadata")
    callback = callback if isinstance(callback, Mapping) else {}
    auxiliary = execution.get("preserved_auxiliary_artifacts")
    auxiliary = auxiliary if isinstance(auxiliary, Mapping) else {}
    artifacts = {
        name: {
            "path": item.get("path"),
            "sha256": item.get("sha256"),
        }
        for name, item in auxiliary.items()
        if name in {"retrieval_provenance", "selection_provenance"}
        and isinstance(item, Mapping)
    }
    return {
        "upstream_reported": deepcopy(
            callback.get("retrieval_provenance")
            if isinstance(callback.get("retrieval_provenance"), (Mapping, list))
            else None
        ),
        "preserved_artifacts": artifacts,
    }


def _evaluator_metadata(
    path: Path | None,
    report: Mapping[str, Any] | None,
    contract: ComparisonProtocol,
) -> dict[str, Any]:
    if path is None or report is None:
        return {
            "policy": contract.as_dict()["evaluator"]["policy"],
            "config_sha256": contract.as_dict()["evaluator"].get(
                "config_sha256"
            ),
            "report": None,
        }
    return {
        "policy": contract.as_dict()["evaluator"]["policy"],
        "config_sha256": contract.as_dict()["evaluator"].get("config_sha256"),
        "entrypoint": "benchmark.api.evaluation.run_evaluate",
        "workflow": report.get("workflow"),
        "profile_version": report.get("evaluation_profile_version")
        or report.get("profile_version"),
        "scoring_spec_version": report.get("scoring_spec_version"),
        "report": path.resolve().as_posix(),
        "report_sha256": _file_sha256(path),
    }


def _selected_iteration(scene: Mapping[str, Any]) -> int | None:
    metadata = scene.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    compatibility = metadata.get("harness_compatibility")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    value = compatibility.get("selected_iteration")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _append_input_immutability_violations(
    validation: dict[str, Any],
    *,
    protocol_path: Path,
    protocol_hash_before: str,
    catalog: CanonicalAssetCatalog | None,
    materialization: Mapping[str, Any] | None,
) -> None:
    actual_protocol = _file_sha256(protocol_path)
    if actual_protocol != protocol_hash_before:
        validation["violations"].append(
            {
                "code": "protocol_mutated",
                "message": "runner mutated the frozen protocol artifact",
                "details": {"before": protocol_hash_before, "after": actual_protocol},
            }
        )
    if catalog is not None and isinstance(materialization, Mapping):
        catalog_path = Path(str(materialization["catalog_path"]))
        if _file_sha256(catalog_path) != materialization["catalog_file_sha256"]:
            validation["violations"].append(
                {
                    "code": "catalog_mutated",
                    "message": "runner mutated the immutable catalog snapshot",
                    "details": {},
                }
            )
        for path_key, hash_key, code in (
            (
                "method_catalog_path",
                "method_payload_file_sha256",
                "method_materialization_mutated",
            ),
            (
                "comparison_control_path",
                "control_file_sha256",
                "comparison_control_mutated",
            ),
            (
                "converter_asset_manifest_path",
                "converter_manifest_file_sha256",
                "converter_manifest_mutated",
            ),
        ):
            path = Path(str(materialization[path_key]))
            if _file_sha256(path) != materialization[hash_key]:
                validation["violations"].append(
                    {
                        "code": code,
                        "message": "runner or conversion mutated a frozen comparison input",
                        "details": {"path": path.resolve().as_posix()},
                    }
                )
    validation["valid_comparison_run"] = not validation["violations"]


def _append_ineligibility(
    report: Mapping[str, Any],
    *,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    result = deepcopy(dict(report))
    result["eligible"] = False
    result["status"] = "INELIGIBLE"
    result.setdefault("reasons", []).append(
        {"code": code, "message": message, "details": details}
    )
    return result


def _copy_adapter_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(value or {})
    for key in ("execution", "comparison_support"):
        if isinstance(config.get(key), Mapping):
            config[key] = dict(config[key])
    return config


def _active_architecture_features(
    generation_input: Mapping[str, Any],
) -> dict[str, Any]:
    contract = generation_input.get("generation_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    architecture = contract.get("architecture")
    architecture = architecture if isinstance(architecture, Mapping) else {}
    physical = architecture.get("physical_walls")
    physical = physical if isinstance(physical, Mapping) else {}
    active_walls = list(physical.get("active_wall_ids") or [])
    active: dict[str, Any] = {}
    if active_walls:
        active["physical_walls"] = active_walls
    for key in ("openings", "rooms", "room_topology"):
        if architecture.get(key):
            active[key] = architecture[key]
    return active


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Native/SharedDB/FrozenAssets comparison case",
    )
    parser.add_argument("--generation-input", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--asset-catalog", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--method-output", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--evaluation-config", default=None)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument(
        "--evaluate-sceneweaver-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    result = run_controlled_generation(
        generation_input=read_json(args.generation_input),
        adapter_name=args.adapter,
        protocol=args.protocol,
        asset_catalog=args.asset_catalog,
        out_dir=args.out_dir,
        adapter_config=(read_json(args.adapter_config) if args.adapter_config else None),
        method_output=args.method_output,
        run_generation=args.run_generation,
        evaluation_kwargs=(
            read_json(args.evaluation_config) if args.evaluation_config else None
        ),
        evaluate_sceneweaver_trajectory=args.evaluate_sceneweaver_trajectory,
    )
    print(f"status: {result['status']}")
    print(f"manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()


__all__ = [
    "COMPARISON_RUN_MANIFEST_SCHEMA_VERSION",
    "ComparisonRunError",
    "run_controlled_generation",
]
