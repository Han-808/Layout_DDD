from __future__ import annotations

import argparse
import http.server
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.experiments import experiment_model_overrides, load_experiment_config, pick_value, resolve_experiment
from benchmark.data import discover_and_normalize_cases
from benchmark.input_modes import canonicalize_input_mode
from benchmark.pipeline import PipelineResources, copy_viewer_assets, load_pipeline_resources, run_case_pipeline
from benchmark.run_config import load_resolved_run_config, pipeline_resources_from_resolved
from benchmark.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one explicit 3D layout benchmark case.")
    parser.add_argument("--experiment_config", default=None, help="Path to a component-composed experiment YAML.")
    parser.add_argument("--experiment", default=None, help="Experiment name from configs/experiment_config.yaml.")
    parser.add_argument("--case", default=None, help="Path to bm_instance JSON.")
    parser.add_argument("--input-mode", "--input_mode", dest="input_mode", default=None, help="Optional model input mode override for this run.")
    parser.add_argument("--model", default=None, help="Model name from configs/model_config.yaml.")
    parser.add_argument("--judge_model", default=None, help="Optional judge model name; defaults to configs/model_config.yaml judge.model, usually 'same'.")
    parser.add_argument("--vlm-judge-input-mode", choices=["json_only", "json_plus_render"], default=None)
    parser.add_argument("--max_repair_iterations", type=int, default=None)
    parser.add_argument("--model_endpoint", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--timeout_seconds", type=int, default=None)
    parser.add_argument("--response_format_json", dest="response_format_json", action="store_true", default=None)
    parser.add_argument("--no_response_format_json", dest="response_format_json", action="store_false")
    parser.add_argument("--out", default=None, help="Output directory for intermediate artifacts.")
    parser.add_argument("--mock_behavior", default=None, help="Optional override for mock behavior.")
    parser.add_argument("--no_viewer_assets", action="store_true", help="Do not copy static viewer files into --out.")
    parser.add_argument("--serve", action="store_true", help="Serve --out with a local HTTP server after the run.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve.")
    args = parser.parse_args()

    cli_model_overrides = _model_cli_overrides(args)
    if args.experiment_config:
        resolved = load_resolved_run_config(
            PROJECT_ROOT,
            experiment_config_path=args.experiment_config,
            experiment_name=args.experiment,
            model_overrides=cli_model_overrides,
        )
        resources = _with_vlm_judge_input_mode(pipeline_resources_from_resolved(PROJECT_ROOT, resolved), args.vlm_judge_input_mode)
        dataset_config = _dataset_config_for_cli(resolved.dataset_config, args.case)
        case_ref, input_json = discover_and_normalize_cases(dataset_config)[0]
        input_json = _apply_input_mode_override(input_json, args.input_mode)
        out_dir = _required(parser, args.out or resolved.data.get("out"), "--out")
        model_name = args.model or resolved.model_name
        judge_model_name = args.judge_model or resolved.data.get("judge_model")
        max_repair_iterations = pick_value(args.max_repair_iterations, resolved.data, "max_repair_iterations")
        state = run_case_pipeline(
            case_path=case_ref.path,
            input_json=input_json,
            out_dir=Path(out_dir),
            model_name=model_name,
            resources=resources,
            max_repair_iterations=max_repair_iterations,
            judge_model_name=judge_model_name,
        )
        if not args.no_viewer_assets and state.get("viewer_scene_path"):
            copy_viewer_assets(out_dir, PROJECT_ROOT)
        print(state["per_case_result_path"])
        if not args.no_viewer_assets and state.get("viewer_scene_path"):
            print(f"viewer: http://127.0.0.1:{args.port}/")
        if args.serve:
            _serve(Path(out_dir), args.port)
        return

    resources = _with_vlm_judge_input_mode(load_pipeline_resources(PROJECT_ROOT), args.vlm_judge_input_mode)
    try:
        experiment = resolve_experiment(load_experiment_config(PROJECT_ROOT), args.experiment)
    except ValueError as exc:
        parser.error(str(exc))
    case_path = _required(parser, pick_value(args.case, experiment, "case"), "--case")
    out_dir = _required(parser, pick_value(args.out, experiment, "out"), "--out")
    model_name = pick_value(args.model, experiment, "model", "mock")
    judge_model_name = pick_value(args.judge_model, experiment, "judge_model")
    max_repair_iterations = pick_value(args.max_repair_iterations, experiment, "max_repair_iterations")
    mock_behavior = pick_value(args.mock_behavior, experiment, "mock_behavior")
    state = run_case_pipeline(
        case_path=Path(case_path),
        input_json=_apply_input_mode_override(read_json(case_path), args.input_mode) if args.input_mode else None,
        out_dir=Path(out_dir),
        model_name=model_name,
        resources=resources,
        max_repair_iterations=max_repair_iterations,
        mock_behavior=mock_behavior,
        model_overrides=_merge_overrides(experiment_model_overrides(experiment), cli_model_overrides),
        judge_model_name=judge_model_name,
    )
    if not args.no_viewer_assets and state.get("viewer_scene_path"):
        copy_viewer_assets(out_dir, PROJECT_ROOT)
    print(state["per_case_result_path"])
    if not args.no_viewer_assets and state.get("viewer_scene_path"):
        print(f"viewer: http://127.0.0.1:{args.port}/")
    if args.serve:
        _serve(Path(out_dir), args.port)


def _serve(out_dir: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


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


def _dataset_config_for_cli(dataset_config: dict, case_override: str | None) -> dict:
    config = dict(dataset_config)
    if case_override:
        config["path"] = case_override
    for key in ["path", "root", "cases_dir", "case", "source_path"]:
        if config.get(key):
            path = Path(config[key])
            config[key] = str(path if path.is_absolute() else PROJECT_ROOT / path)
            break
    return config


def _apply_input_mode_override(input_json: dict, input_mode: str | None) -> dict:
    if not input_mode:
        return input_json
    mode = canonicalize_input_mode(input_mode)
    patched = dict(input_json)
    patched["scene_representation_mode"] = mode
    source = patched.get("source")
    if isinstance(source, dict):
        source = dict(source)
        source["input_representation_mode"] = mode
        source["scene_representation_mode"] = mode
        patched["source"] = source
    return patched


def _with_vlm_judge_input_mode(resources: PipelineResources, mode: str | None) -> PipelineResources:
    if not mode:
        return resources
    benchmark_config = deepcopy(resources.benchmark_config)
    benchmark_config["vlm_judge_input_mode"] = mode
    evaluation = dict(benchmark_config.get("evaluation") or {})
    evaluation["vlm_judge_input_mode"] = mode
    benchmark_config["evaluation"] = evaluation
    return PipelineResources(
        model_config=resources.model_config,
        benchmark_config=benchmark_config,
        layout_schema=resources.layout_schema,
        scene_schema=resources.scene_schema,
        resolved_run_config=resources.resolved_run_config,
    )

if __name__ == "__main__":
    main()
