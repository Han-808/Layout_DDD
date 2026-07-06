from __future__ import annotations

import shutil
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path

from benchmark.data import iter_case_paths
from benchmark.metrics.aggregate import aggregate_case_results, write_benchmark_metrics_outputs
from benchmark.models import create_model
from benchmark.input_modes import canonicalize_input_mode
from benchmark.utils.io import load_json_schema, load_yaml, read_json
from benchmark.workflow import run_workflow
from benchmark.workflow.artifacts import configured_max_repair_iterations, output_settings
from benchmark.workflow.judge_evidence_selector import judge_generation_overrides
from benchmark.workflow.state import BenchmarkState


MODEL_OVERRIDE_KEYS = {
    "model_id": "model",
    "model_endpoint": "endpoint",
}
SAME_MODEL = "same"


@dataclass(frozen=True)
class PipelineResources:
    model_config: dict
    benchmark_config: dict
    layout_schema: dict
    resolved_run_config: dict | None = None


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
    model_overrides: dict | None = None,
    judge_model_name: str | None = None,
    input_json: dict | None = None,
) -> BenchmarkState:
    model_config = deepcopy(resources.model_config)
    if mock_behavior:
        model_config.setdefault("models", {}).setdefault("mock", {})["behavior"] = mock_behavior
    apply_model_overrides(model_config, model_name, model_overrides)

    model = create_model(model_name, model_config)
    resolved_judge_model_name, judge_model = create_judge_model(
        model_name=model_name,
        model=model,
        model_config=model_config,
        judge_model_name=judge_model_name,
        benchmark_config=resources.benchmark_config,
    )
    repair_budget = (
        int(max_repair_iterations)
        if max_repair_iterations is not None
        else configured_max_repair_iterations(resources.benchmark_config)
    )
    benchmark_config = deepcopy(resources.benchmark_config)
    resolved_run_config = deepcopy(resources.resolved_run_config) if resources.resolved_run_config else None
    if resolved_run_config:
        benchmark_config.setdefault("config_refs", resolved_run_config.get("config_refs", {}))
        benchmark_config.setdefault("config_hash", resolved_run_config.get("config_hash", ""))

    return run_workflow(
        {
            "case_path": str(Path(case_path)),
            "out_dir": str(Path(out_dir)),
            **({"input_json": input_json} if input_json is not None else {}),
            "model": model,
            "model_name": model_name,
            "judge_model": judge_model,
            "judge_model_name": resolved_judge_model_name,
            "layout_schema": resources.layout_schema,
            "benchmark_config": benchmark_config,
            "resolved_run_config": resolved_run_config,
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
    model_overrides: dict | None = None,
    judge_model_name: str | None = None,
    cases: list[tuple[Path, dict]] | None = None,
    input_modes: list[str] | None = None,
) -> tuple[dict, Path, Path]:
    root_out = Path(out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    case_metrics = []
    discovered_cases = cases or [(Path(case_path), None) for case_path in iter_case_paths(cases_dir)]
    expanded_cases = _expand_case_modes(discovered_cases, input_modes)
    for case_path, input_json, mode in expanded_cases:
        case_out = root_out / _case_output_id(case_path, input_json=input_json, mode=mode)
        state = run_case_pipeline(
            case_path=case_path,
            out_dir=case_out,
            model_name=model_name,
            resources=resources,
            max_repair_iterations=max_repair_iterations,
            mock_behavior=mock_behavior,
            model_overrides=model_overrides,
            judge_model_name=judge_model_name,
            input_json=input_json,
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


def apply_model_overrides(model_config: dict, model_name: str, model_overrides: dict | None) -> None:
    if not model_overrides:
        return
    selected = model_config.setdefault("models", {}).setdefault(model_name, {})
    for key, value in model_overrides.items():
        if value is None:
            continue
        config_key = MODEL_OVERRIDE_KEYS.get(key, key)
        selected[config_key] = value


def create_judge_model(
    *,
    model_name: str,
    model: object,
    model_config: dict,
    judge_model_name: str | None = None,
    benchmark_config: dict | None = None,
) -> tuple[str, object]:
    resolved = judge_model_name or _configured_judge_model_name(model_config)
    if resolved in {"", SAME_MODEL, "same_model"}:
        return model_name, _model_with_judge_generation_overrides(model, benchmark_config)
    return resolved, _model_with_judge_generation_overrides(create_model(resolved, model_config), benchmark_config)


def _configured_judge_model_name(model_config: dict) -> str:
    judge_config = model_config.get("judge")
    if isinstance(judge_config, dict):
        return str(judge_config.get("model") or SAME_MODEL)
    return SAME_MODEL


def _model_with_judge_generation_overrides(model: object, benchmark_config: dict | None) -> object:
    runtime_profile = getattr(model, "runtime_profile", None)
    overrides = judge_generation_overrides(benchmark_config, runtime_profile)
    if not overrides:
        return model
    judge_model = copy(model)
    for key, value in overrides.items():
        if value is not None and hasattr(judge_model, key):
            setattr(judge_model, key, value)
    return judge_model


def copy_viewer_assets(out_dir: str | Path, project_root: str | Path) -> None:
    viewer_src = Path(project_root) / "web" / "viewer"
    for path in viewer_src.iterdir():
        if path.is_file():
            shutil.copy2(path, Path(out_dir) / path.name)


def _expand_case_modes(cases: list[tuple[Path, dict | None]], input_modes: list[str] | None) -> list[tuple[Path, dict | None, str | None]]:
    if not input_modes:
        return [(case_path, input_json, None) for case_path, input_json in cases]
    modes = [canonicalize_input_mode(mode) for mode in input_modes]
    expanded = []
    for case_path, input_json in cases:
        base = input_json if isinstance(input_json, dict) else read_json(case_path)
        if not isinstance(base, dict):
            raise ValueError(f"Case at {case_path} must be a JSON object.")
        for mode in modes:
            expanded.append((case_path, _case_with_input_mode(base, mode), mode))
    return expanded


def _case_with_input_mode(case: dict, mode: str) -> dict:
    patched = deepcopy(case)
    patched["scene_representation_mode"] = mode
    source = patched.get("source")
    if isinstance(source, dict):
        source = dict(source)
        source["input_representation_mode"] = mode
        source["scene_representation_mode"] = mode
        patched["source"] = source
    return patched


def _case_output_id(case_path: str | Path, *, input_json: dict | None = None, mode: str | None = None) -> str:
    path = Path(case_path)
    case = input_json
    if case is None:
        try:
            case = read_json(path)
        except (OSError, ValueError):
            case = {}
    base = str(case.get("case_id") or case.get("task_id") or path.stem) if isinstance(case, dict) else path.stem
    return f"{base}__{mode}" if mode else base
