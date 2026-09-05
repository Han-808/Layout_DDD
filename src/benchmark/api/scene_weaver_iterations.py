"""Offline evaluation of a preserved native SceneWeaver trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Mapping

from benchmark.adapters import get_adapter
from benchmark.adapters.common.execution import artifact_sha256
from benchmark.adapters.scene_weaver.converter import discover_layout_iterations
from benchmark.api.evaluation import run_evaluate
from benchmark.generation_comparison.evaluation_runtime import (
    CanonicalEvaluationRuntime, runtime_evaluation_options,
)
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import read_json, write_json


ITERATION_SUMMARY_SCHEMA_VERSION = "sceneweaver_iteration_evaluation_v1"


def evaluate_scene_weaver_iterations(
    *,
    native_output: str | Path,
    generation_input: dict,
    out_dir: str | Path,
    adapter_config: Mapping[str, Any] | None = None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    evaluation_runtime: CanonicalEvaluationRuntime | None = None,
) -> dict[str, Any]:
    """Convert and evaluate every native layout with the canonical evaluator."""

    validate_generation_input(generation_input)
    native_root = Path(native_output).expanduser().resolve()
    iterations = discover_layout_iterations(native_root)
    if not iterations:
        raise ArtifactValidationError(
            "SceneWeaver native trajectory contains no layout_<iteration>.json"
        )
    evaluation_options = dict(evaluation_kwargs or {})
    forbidden = sorted(
        key
        for key in ("scene", "out", "evaluation_mode")
        if key in evaluation_options
    )
    if forbidden:
        raise ArtifactValidationError(
            "SceneWeaver iteration evaluation controls the canonical route; "
            f"remove evaluator arguments {forbidden}"
        )
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    native_digest_before, _ = artifact_sha256(native_root)
    base_config = dict(adapter_config or {})
    for key in ("layout_path", "selected_iteration", "iteration_selection_policy"):
        base_config.pop(key, None)

    rows: list[dict[str, Any]] = []
    previous_score: float | None = None
    workflows: set[str] = set()
    for iteration, native_layout in sorted(iterations.items()):
        iteration_dir = output_root / "iterations" / f"iteration_{iteration:03d}"
        canonical_dir = iteration_dir / "canonical"
        adapter = get_adapter("scene_weaver")
        config = {**base_config, "selected_iteration": iteration}
        canonical_path = adapter.materialize_output(
            native_root,
            generation_input,
            canonical_dir,
            config=config,
        )
        converter_metadata_path = write_json(
            iteration_dir / "converter_metadata.json",
            getattr(adapter, "last_parse_metadata", {}),
        )
        scene = read_json(canonical_path)
        report_path = iteration_dir / "evaluation_report.json"
        report = run_evaluate(
            scene=scene,
            out=report_path,
            **runtime_evaluation_options(evaluation_options, evaluation_runtime,
                                         scene=scene, out_dir=iteration_dir / "evaluation_runtime"),
        )
        workflows.add(str(report.get("workflow") or ""))
        score_value = report.get("benchmark_score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        delta = (
            score - previous_score
            if score is not None and previous_score is not None
            else None
        )
        if score is not None:
            previous_score = score
        native_layout_digest, _ = artifact_sha256(native_layout)
        rows.append(
            {
                "iteration": iteration,
                "native_artifact": native_layout.resolve().as_posix(),
                "native_artifact_sha256": native_layout_digest,
                "related_native_artifacts": _related_iteration_artifacts(
                    native_root,
                    native_layout,
                    iteration,
                ),
                "canonical_scene": canonical_path.resolve().as_posix(),
                "converter_metadata": converter_metadata_path.resolve().as_posix(),
                "evaluation_report": report_path.resolve().as_posix(),
                "evaluation_workflow": report.get("workflow"),
                "benchmark_score": score_value,
                "benchmark_score_100": report.get("benchmark_score_100"),
                "benchmark_score_status": report.get("benchmark_score_status"),
                "delta_benchmark_score": delta,
            }
        )

    native_digest_after, _ = artifact_sha256(native_root)
    if native_digest_after != native_digest_before:
        raise ArtifactValidationError(
            "SceneWeaver native trajectory changed during offline evaluation"
        )
    summary = {
        "schema_version": ITERATION_SUMMARY_SCHEMA_VERSION,
        "harness": "scene_weaver",
        "native_trajectory": native_root.as_posix(),
        "native_trajectory_sha256": native_digest_before,
        "native_trajectory_verified_unchanged": True,
        "available_iterations": sorted(iterations),
        "evaluation_workflows": sorted(workflows),
        "benchmark_feedback_used_by_native_loop": False,
        "iterations": rows,
    }
    summary_path = write_json(output_root / "iteration_summary.json", summary)
    return {**summary, "summary_path": summary_path.resolve().as_posix()}


def _related_iteration_artifacts(
    native_root: Path,
    native_layout: Path,
    iteration: int,
) -> list[str]:
    root = native_root if native_root.is_dir() else native_root.parent
    pattern = re.compile(rf"(?:^|_){iteration}(?:_|\.|$)")
    return sorted(
        path.resolve().as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != native_layout
        and pattern.search(path.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every preserved SceneWeaver native iteration",
    )
    parser.add_argument("--native-output", required=True)
    parser.add_argument("--generation-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--evaluation-config", default=None)
    args = parser.parse_args()
    result = evaluate_scene_weaver_iterations(
        native_output=args.native_output,
        generation_input=read_json(args.generation_input),
        out_dir=args.out_dir,
        adapter_config=(
            read_json(args.adapter_config) if args.adapter_config else None
        ),
        evaluation_kwargs=(
            read_json(args.evaluation_config) if args.evaluation_config else None
        ),
    )
    print(f"iterations: {len(result['iterations'])}")
    print(f"summary: {result['summary_path']}")


if __name__ == "__main__":
    main()


__all__ = [
    "ITERATION_SUMMARY_SCHEMA_VERSION",
    "evaluate_scene_weaver_iterations",
]
