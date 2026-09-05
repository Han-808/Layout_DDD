"""Append-only evaluation recovery: reuse a validated generation, never rerun it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.adapters.common.execution import artifact_sha256, redact_private_locators
from benchmark.api.evaluation import run_evaluate
from benchmark.generation_comparison.catalog import load_asset_catalog
from benchmark.generation_comparison.evaluation_acceptance import evaluate_report_acceptance
from benchmark.generation_comparison.evaluation_runtime import (
    CanonicalEvaluationRuntime, runtime_evaluation_options,
)
from benchmark.generation_comparison.prepared import verify_prepared_artifacts
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


def reevaluate_prepared_unit(
    *, prepared_dir: str | Path, case_id: str, method: str, out_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the same preserved canonical scene with the same frozen policy.

    This calls the real evaluator and may spend Judge/camera API calls. It is
    NOT a no-call preflight. No generator, asset selector or converter is used.
    SceneWeaver here means the originally selected final state, not a new loop.
    """
    root = Path(prepared_dir).expanduser().resolve()
    manifest = read_json(root / "pilot_manifest.json")
    verified = verify_prepared_artifacts(root, manifest)
    if case_id not in verified["cases"] or method not in manifest["methods"]:
        raise ArtifactValidationError("unknown prepared method/case")
    unit = root / "cases" / case_id / method
    proof_path = unit / "comparison/generation_manifest.json"
    proof = read_json(proof_path)
    current = read_json(unit / "comparison/run_manifest.json")
    case = verified["cases"][case_id]
    if (proof.get("valid_comparison_run") is not True
            or current.get("valid_comparison_run") is not True
            or proof.get("method") != method or proof.get("case_id") != case_id
            or proof.get("protocol_sha256") != case["protocol"].sha256):
        raise ArtifactValidationError("source generation is not a valid matching comparison unit")
    destination = Path(out_dir).expanduser().resolve()
    if destination.is_relative_to(root) or destination.exists():
        raise FileExistsError("reevaluation requires a fresh directory outside the original prepared run")

    source_files = {}
    for name in ("native_artifact", "canonical_scene", "evaluation_input"):
        path = Path(str(proof.get(name) or "")).resolve()
        if not path.is_relative_to(unit) or not path.exists():
            raise ArtifactValidationError(f"preserved {name} missing/outside original unit")
        actual, _ = artifact_sha256(path)
        if actual != proof.get(f"{name}_sha256"):
            raise ArtifactValidationError(f"preserved {name} hash mismatch")
        source_files[path] = actual
    source_files[proof_path] = artifact_sha256(proof_path)[0]
    source_files[unit / "comparison/run_manifest.json"] = artifact_sha256(unit / "comparison/run_manifest.json")[0]
    scene = read_json(proof["canonical_scene"])
    options = read_json(proof["evaluation_input"])
    policy = verified["evaluator_policy"]
    expected_options = {**policy.get("static_kwargs", {}),
                        "scene_request": case["generation_input"]["scene_request"],
                        "object_plan": case["object_plan"]}
    if options != expected_options:
        raise ArtifactValidationError("preserved evaluator inputs differ from the prepared policy/case")
    if not isinstance(policy.get("runtime"), dict):
        raise ArtifactValidationError("prepared evaluator has no recoverable production runtime")
    runtime = CanonicalEvaluationRuntime(policy["runtime"])
    destination.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": "controlled_generation_reevaluation_v1",
        "status": "EVALUATING", "method": method, "case_id": case_id,
        "source_generation_manifest": proof_path.as_posix(),
        "source_generation_manifest_sha256": source_files[proof_path],
        "canonical_scene": proof["canonical_scene"],
        "canonical_scene_sha256": proof["canonical_scene_sha256"],
        "native_artifact": proof["native_artifact"],
        "native_artifact_sha256": proof["native_artifact_sha256"],
        "protocol_sha256": proof["protocol_sha256"],
        "catalog": verified["catalog"].identity,
        "evaluator_config_sha256": manifest["evaluator_config_sha256"],
        "generation_reexecuted": False, "converter_reexecuted": False,
        "benchmark_feedback_to_generator": False,
        "sceneweaver_trajectory_reevaluated": False,
    }
    result_path = destination / "reevaluation_manifest.json"
    write_json(result_path, result)
    try:
        report_path = destination / "evaluation_report.json"
        report = run_evaluate(scene=scene, out=report_path,
                              **runtime_evaluation_options(options, runtime, scene=scene,
                                                           out_dir=destination / "evaluation_runtime"))
        for path, digest in source_files.items():
            if artifact_sha256(path)[0] != digest:
                raise ArtifactValidationError("reevaluation changed a preserved source artifact")
        # Also detect renderer-side asset mutation; the catalog contains exact
        # per-mesh byte hashes, not only category/size declarations.
        load_asset_catalog(verified["catalog"].as_dict(), hash_local_meshes=True)
        acceptance = evaluate_report_acceptance(report, policy)
        result.update({
            "status": "COMPLETED" if acceptance["accepted"] else "INCOMPLETE_EVALUATION",
            "evaluation_acceptance": acceptance,
            "evaluation_report": report_path.as_posix(),
            "evaluation_report_sha256": artifact_sha256(report_path)[0],
            "benchmark_score": report.get("benchmark_score"),
            "benchmark_score_status": report.get("benchmark_score_status"),
            "source_artifacts_verified_unchanged": True,
        })
    except BaseException as exc:
        result.update({"status": "CANCELLED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED",
                       "error": {"type": type(exc).__name__, "message": redact_private_locators(str(exc))}})
        write_json(result_path, result)
        raise
    write_json(result_path, result)
    return {**result, "manifest_path": result_path.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = reevaluate_prepared_unit(**vars(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
