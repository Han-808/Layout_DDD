from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.experiments import experiment_model_overrides, load_experiment_config, pick_value, resolve_experiment
from benchmark.data import discover_and_normalize_cases
from benchmark.input_modes import list_main_input_modes
from benchmark.pipeline import load_pipeline_resources, run_benchmark_pipeline
from benchmark.run_config import load_resolved_run_config, pipeline_resources_from_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a folder of explicit 3D layout benchmark cases.")
    parser.add_argument("--experiment_config", default=None, help="Path to a component-composed experiment YAML.")
    parser.add_argument("--experiment", default=None, help="Experiment name from configs/experiment_config.yaml.")
    parser.add_argument("--cases", default=None, help="Directory containing bm_instance JSON files.")
    parser.add_argument(
        "--input-modes",
        "--input_modes",
        dest="input_modes",
        nargs="*",
        default=None,
        help="Optional comma-separated or space-separated input modes. When set, each case is run once per mode.",
    )
    parser.add_argument("--model", default=None, help="Model name from configs/model_config.yaml.")
    parser.add_argument("--judge_model", default=None, help="Optional judge model name; defaults to configs/model_config.yaml judge.model, usually 'same'.")
    parser.add_argument("--max_repair_iterations", type=int, default=None)
    parser.add_argument("--model_endpoint", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--timeout_seconds", type=int, default=None)
    parser.add_argument("--response_format_json", dest="response_format_json", action="store_true", default=None)
    parser.add_argument("--no_response_format_json", dest="response_format_json", action="store_false")
    parser.add_argument("--out", default=None, help="Output directory for benchmark artifacts.")
    parser.add_argument("--mock_behavior", default=None, help="Optional override for mock behavior.")
    args = parser.parse_args()

    cli_model_overrides = _model_cli_overrides(args)
    if args.experiment_config:
        resolved = load_resolved_run_config(
            PROJECT_ROOT,
            experiment_config_path=args.experiment_config,
            experiment_name=args.experiment,
            model_overrides=cli_model_overrides,
        )
        resources = pipeline_resources_from_resolved(PROJECT_ROOT, resolved)
        dataset_config = _dataset_config_for_cli(resolved.dataset_config, args.cases)
        cases = [(case_ref.path, input_json) for case_ref, input_json in discover_and_normalize_cases(dataset_config)]
        out_dir = _required(parser, args.out or resolved.data.get("out"), "--out")
        _, json_path, csv_path = run_benchmark_pipeline(
            cases_dir=Path(dataset_config.get("path") or dataset_config.get("root") or "."),
            cases=cases,
            out_dir=Path(out_dir),
            model_name=args.model or resolved.model_name,
            resources=resources,
            max_repair_iterations=pick_value(args.max_repair_iterations, resolved.data, "max_repair_iterations"),
            mock_behavior=args.mock_behavior,
            judge_model_name=args.judge_model or resolved.data.get("judge_model"),
            input_modes=_parse_input_modes(args.input_modes),
        )
        print(json_path)
        print(csv_path)
        return

    resources = load_pipeline_resources(PROJECT_ROOT)
    try:
        experiment = resolve_experiment(load_experiment_config(PROJECT_ROOT), args.experiment)
    except ValueError as exc:
        parser.error(str(exc))
    cases_dir = _required(parser, pick_value(args.cases, experiment, "cases"), "--cases")
    out_dir = _required(parser, pick_value(args.out, experiment, "out"), "--out")
    model_name = pick_value(args.model, experiment, "model", "mock")
    judge_model_name = pick_value(args.judge_model, experiment, "judge_model")
    max_repair_iterations = pick_value(args.max_repair_iterations, experiment, "max_repair_iterations")
    mock_behavior = pick_value(args.mock_behavior, experiment, "mock_behavior")
    _, json_path, csv_path = run_benchmark_pipeline(
        cases_dir=Path(cases_dir),
        out_dir=Path(out_dir),
        model_name=model_name,
        resources=resources,
        max_repair_iterations=max_repair_iterations,
        mock_behavior=mock_behavior,
        model_overrides=_merge_overrides(experiment_model_overrides(experiment), cli_model_overrides),
        judge_model_name=judge_model_name,
        input_modes=_parse_input_modes(args.input_modes),
    )
    print(json_path)
    print(csv_path)


def _required(parser: argparse.ArgumentParser, value: object, flag: str) -> str:
    if value in {None, ""}:
        parser.error(f"{flag} is required unless provided by --experiment")
    return str(value)


def _model_cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "model_endpoint": args.model_endpoint,
        "model_id": args.model_id,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "response_format_json": args.response_format_json,
    }


def _merge_overrides(base: dict | None, patch: dict | None) -> dict | None:
    merged = dict(base or {})
    for key, value in (patch or {}).items():
        if value is not None:
            merged[key] = value
    return merged or None


def _dataset_config_for_cli(dataset_config: dict, cases_override: str | None) -> dict:
    config = dict(dataset_config)
    if cases_override:
        config["path"] = cases_override
    for key in ["path", "root", "cases_dir", "case", "source_path"]:
        if config.get(key):
            path = Path(config[key])
            config[key] = str(path if path.is_absolute() else PROJECT_ROOT / path)
            break
    return config


def _parse_input_modes(value: list[str] | None) -> list[str] | None:
    if value is None:
        return None
    text = " ".join(value).strip()
    if not text:
        return None
    if text == "main":
        return list_main_input_modes()
    modes = []
    for chunk in text.replace(",", " ").split():
        if chunk:
            modes.append(chunk)
    return modes or None

if __name__ == "__main__":
    main()
