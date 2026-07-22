"""Canonical generation/materialization API and CLI implementation.

Summary:
    Turns a method's submission into a canonical scene by dispatching it through
    a generation adapter. It normalizes and validates; it never evaluates.

Input:
    - A canonical ``generation_input`` (or a method's raw ``--method-output``).
    - ``--adapter`` (e.g. ``layout_json`` / ``object_state``) and optional
      ``--adapter-config``.
    - ``--out-dir`` for artifacts.

Output:
    - A canonical ``generated_scene`` JSON (O1/O3 object state) plus per-stage
      generation artifacts written under ``--out-dir``.

Function:
    Selects the adapter, converts the submitted format into canonical scene
    state, and applies schema validation. Adapters may bind IDs, transform
    coordinates/field names, and dereference an explicitly selected asset, but
    must not invent objects, assets, or scene semantics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from benchmark.adapters import get_adapter
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import build_direct_natural_language_generation_input
from benchmark.utils.io import read_json, write_json


def run_generate(
    *,
    generation_input: dict,
    adapter_name: str,
    out_dir: str | Path,
    method_output: str | Path | None = None,
    adapter_config: dict | None = None,
    run_generation: bool = False,
    evaluation_report: dict | None = None,
    previous_generated_scene: dict | None = None,
    iteration: int | None = None,
) -> dict:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_dir = output_dir / "generator"
    generator_dir.mkdir(parents=True, exist_ok=True)
    prepared_generation_input = attach_self_reflection_feedback(
        generation_input,
        evaluation_report=evaluation_report,
        previous_generated_scene=previous_generated_scene,
        iteration=iteration,
    )
    adapter = get_adapter(adapter_name)
    config = adapter_config or {}
    io_contract = adapter.resolve_io_contract(prepared_generation_input, config=config)
    method_input_path = adapter.prepare_input(prepared_generation_input, generator_dir, config=config)
    generated_scene_path: Path | None = None

    if method_output:
        generated_scene_path = adapter.materialize_output(
            Path(method_output),
            prepared_generation_input,
            output_dir,
            config=config,
            execution_dir=generator_dir,
        )
        status = {"status": "generated_scene_available", "reason": "native method output was provided", "generated_scene": generated_scene_path.name}
    elif run_generation:
        method_output_path = adapter.run_generation(method_input_path, generator_dir, config=config)
        generated_scene_path = adapter.materialize_output(
            method_output_path,
            prepared_generation_input,
            output_dir,
            config=config,
            execution_dir=generator_dir,
        )
        status = {"status": "generated_scene_available", "reason": "adapter generation completed", "generated_scene": generated_scene_path.name}
    else:
        status = {
            "status": "generation_skipped",
            "reason": "No method output provided and --run-generation was not set.",
            "next_expected_input": "method_output",
        }

    workflow_status_path = write_json(output_dir / "workflow_status.json", status)
    metadata = {
        "adapter": adapter.name,
        "adapter_capabilities": adapter.capabilities.as_dict(),
        "io_contract": io_contract.as_dict(),
        "generator_output_schema": getattr(adapter, "output_schema", None),
        "method_input_path": method_input_path.as_posix(),
        "generator_dir": generator_dir.as_posix(),
        "generated_scene_path": generated_scene_path.as_posix() if generated_scene_path else None,
        "run_generation": bool(run_generation),
        "provided_method_output": str(method_output) if method_output else None,
        "generation_run": getattr(adapter, "last_run_metadata", None),
        "materialization": getattr(adapter, "last_materialization_metadata", None),
        "self_reflection": {
            "enabled": evaluation_report is not None,
            "iteration": iteration,
        },
    }
    metadata_path = write_json(output_dir / "adapter_metadata.json", metadata)
    return {
        "adapter": adapter.name,
        "method_input": method_input_path.as_posix(),
        "generated_scene": generated_scene_path.as_posix() if generated_scene_path else None,
        "workflow_status": workflow_status_path.as_posix(),
        "adapter_metadata": metadata_path.as_posix(),
        "status": status,
    }


def run_generate_from_natural_language(
    *,
    instruction: str,
    scene_type: str,
    room: dict,
    request_id: str,
    adapter_name: str,
    out_dir: str | Path,
    method_output: str | Path | None = None,
    evaluator_output_type: str = O1_OBJECT_STATE,
    adapter_config: dict | None = None,
    run_generation: bool = False,
) -> dict:
    """Interface-only entry point for generators that expect natural language."""

    generation_input = build_direct_natural_language_generation_input(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=room,
        evaluator_output_type=evaluator_output_type,
    )
    return run_generate(
        generation_input=generation_input,
        adapter_name=adapter_name,
        out_dir=out_dir,
        method_output=method_output,
        adapter_config=adapter_config,
        run_generation=run_generation,
    )


def attach_self_reflection_feedback(
    generation_input: dict,
    *,
    evaluation_report: dict | None = None,
    previous_generated_scene: dict | None = None,
    iteration: int | None = None,
) -> dict:
    """Return generation input augmented with evaluator feedback for retry attempts.

    The current harness treats this as an interface contract only. Adapters may
    ignore the field, pass it directly to an external generator, or translate it
    into method-specific repair prompts.
    """

    if evaluation_report is None:
        return generation_input
    updated = deepcopy(generation_input)
    updated["self_reflection"] = {
        "enabled": True,
        "iteration": int(iteration or 0),
        "source": "evaluate.py",
        "target": "generate.py",
        "previous_evaluation": evaluation_report,
        "previous_generated_scene": previous_generated_scene,
    }
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch canonical generation_input.json through a generation adapter.")
    parser.add_argument("--generation-input", required=True)
    parser.add_argument("--adapter", default="layout_json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--method-output",
        default=None,
        help="Native generator output: O1 JSON, O2 program, or O3 package.",
    )
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--generator-endpoint", default=None)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--generator-api-key-env", default=None)
    parser.add_argument(
        "--generator-max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=None,
    )
    parser.add_argument(
        "--generator-send-temperature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--evaluation-report", default=None, help="Optional evaluate.py output to attach as self-reflection feedback.")
    parser.add_argument("--previous-generated-scene", default=None, help="Optional previous generated_scene.json to attach with self-reflection feedback.")
    parser.add_argument("--iteration", type=int, default=None, help="Self-reflection iteration index for adapter metadata and input.")
    args = parser.parse_args()

    generation_input = read_json(_path_arg(args.generation_input))
    adapter_config = read_json(_path_arg(args.adapter_config)) if args.adapter_config else None
    if adapter_config is not None and not isinstance(adapter_config, dict):
        parser.error("--adapter-config must point to a JSON object.")
    adapter_config = adapter_config or {}
    for key, value in {
        "endpoint": args.generator_endpoint,
        "model": args.generator_model,
        "api_key_env": args.generator_api_key_env,
        "max_tokens_field": args.generator_max_tokens_field,
        "send_temperature": args.generator_send_temperature,
    }.items():
        if value is not None:
            adapter_config[key] = value
    result = run_generate(
        generation_input=generation_input,
        adapter_name=args.adapter,
        out_dir=_path_arg(args.out_dir),
        method_output=_path_arg(args.method_output) if args.method_output else None,
        adapter_config=adapter_config or None,
        run_generation=args.run_generation,
        evaluation_report=read_json(_path_arg(args.evaluation_report)) if args.evaluation_report else None,
        previous_generated_scene=read_json(_path_arg(args.previous_generated_scene)) if args.previous_generated_scene else None,
        iteration=args.iteration,
    )
    print(f"status: {result['status']['status']}")
    print(f"method_input: {result['method_input']}")
    if result.get("generated_scene"):
        print(f"generated_scene: {result['generated_scene']}")
    print(f"workflow_status: {result['workflow_status']}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
