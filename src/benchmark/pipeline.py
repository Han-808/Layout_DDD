from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from benchmark.data import iter_case_paths
from benchmark.metrics.aggregate import aggregate_case_results, write_benchmark_metrics_outputs
from benchmark.models import create_model
from benchmark.utils.io import load_json_schema, load_yaml, read_json
from benchmark.workflow import run_workflow
from benchmark.workflow.artifacts import configured_max_repair_iterations, output_settings
from benchmark.workflow.state import BenchmarkState


@dataclass(frozen=True)
class PipelineResources:
    model_config: dict
    benchmark_config: dict
    layout_schema: dict


def load_pipeline_resources(project_root: str | Path) -> PipelineResources:
    root = Path(project_root)
    return PipelineResources(
        model_config=load_yaml(root / "configs" / "model_config.yaml", default={}),
        benchmark_config=load_yaml(root / "configs" / "benchmark_config.yaml", default={}),
        layout_schema=load_json_schema(root / "schemas" / "layout.schema.json"),
    )


def run_case_pipeline(
    *,
    case_path: str | Path,
    out_dir: str | Path,
    model_name: str = "mock",
    resources: PipelineResources,
    max_repair_iterations: int | None = None,
    mock_behavior: str | None = None,
) -> BenchmarkState:
    model_config = deepcopy(resources.model_config)
    if mock_behavior:
        model_config.setdefault("models", {}).setdefault("mock", {})["behavior"] = mock_behavior

    model = create_model(model_name, model_config)
    repair_budget = (
        int(max_repair_iterations)
        if max_repair_iterations is not None
        else configured_max_repair_iterations(resources.benchmark_config)
    )
    return run_workflow(
        {
            "case_path": str(Path(case_path)),
            "out_dir": str(Path(out_dir)),
            "model": model,
            "model_name": model_name,
            "layout_schema": resources.layout_schema,
            "benchmark_config": resources.benchmark_config,
            "max_repair_iterations": repair_budget,
        }
    )


def run_benchmark_pipeline(
    *,
    cases_dir: str | Path,
    out_dir: str | Path,
    model_name: str = "mock",
    resources: PipelineResources,
    max_repair_iterations: int | None = None,
    mock_behavior: str | None = None,
) -> tuple[dict, Path, Path]:
    root_out = Path(out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    case_metrics = []
    for case_path in iter_case_paths(cases_dir):
        case_out = root_out / _case_output_id(case_path)
        state = run_case_pipeline(
            case_path=case_path,
            out_dir=case_out,
            model_name=model_name,
            resources=resources,
            max_repair_iterations=max_repair_iterations,
            mock_behavior=mock_behavior,
        )
        case_metrics.append(state["case_metrics"])

    summary = aggregate_case_results(case_metrics)
    csv_path, summary_path = write_benchmark_metrics_outputs(case_metrics, root_out)

    outputs = output_settings(resources.benchmark_config)
    legacy_json_name = outputs.get("per_model_summary_json")
    legacy_csv_name = outputs.get("per_model_summary_csv")
    if legacy_json_name and legacy_json_name != "benchmark_summary.json":
        from benchmark.utils.io import write_json

        write_json(root_out / legacy_json_name, summary)
    if legacy_csv_name and legacy_csv_name != "benchmark_metrics.csv":
        (root_out / legacy_csv_name).write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return summary, summary_path, csv_path


def copy_viewer_assets(out_dir: str | Path, project_root: str | Path) -> None:
    viewer_src = Path(project_root) / "web" / "viewer"
    for path in viewer_src.iterdir():
        if path.is_file():
            shutil.copy2(path, Path(out_dir) / path.name)


def _case_output_id(case_path: str | Path) -> str:
    path = Path(case_path)
    try:
        case = read_json(path)
    except (OSError, ValueError):
        return path.stem
    return str(case.get("case_id") or case.get("task_id") or path.stem)
