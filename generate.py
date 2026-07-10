from __future__ import annotations

import argparse
from copy import deepcopy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.adapters import get_adapter
from benchmark.nl_scene.generation_input import build_direct_natural_language_generation_input
from benchmark.utils.io import read_json, write_json


def run_generate(
    *,
    generation_input: dict,
    adapter_name: str,
    out_dir: str | Path,
    generated_scene: str | Path | None = None,
    adapter_config: dict | None = None,
    run_generation: bool = False,
    evaluation_report: dict | None = None,
    previous_generated_scene: dict | None = None,
    iteration: int | None = None,
) -> dict:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_generation_input = attach_self_reflection_feedback(
        generation_input,
        evaluation_report=evaluation_report,
        previous_generated_scene=previous_generated_scene,
        iteration=iteration,
    )
    adapter = get_adapter(adapter_name)
    config = adapter_config or {}
    method_input_path = adapter.prepare_input(prepared_generation_input, output_dir, config=config)
    generated_scene_path: Path | None = None

    if generated_scene:
        generated_scene_path = adapter.parse_output(Path(generated_scene), prepared_generation_input, output_dir, config=config)
        status = {"status": "generated_scene_available", "reason": "generated scene was provided", "generated_scene": generated_scene_path.name}
    elif run_generation:
        method_output_path = adapter.run_generation(method_input_path, output_dir, config=config)
        generated_scene_path = adapter.parse_output(method_output_path, prepared_generation_input, output_dir, config=config)
        status = {"status": "generated_scene_available", "reason": "adapter generation completed", "generated_scene": generated_scene_path.name}
    else:
        status = {
            "status": "generation_skipped",
            "reason": "No generated scene provided and --run-generation was not set.",
            "next_expected_input": "generated_scene.json",
        }

    workflow_status_path = write_json(output_dir / "workflow_status.json", status)
    metadata = {
        "adapter": adapter.name,
        "adapter_capabilities": adapter.capabilities.as_dict(),
        "generator_output_schema": getattr(adapter, "output_schema", None),
        "method_input_path": method_input_path.as_posix(),
        "generated_scene_path": generated_scene_path.as_posix() if generated_scene_path else None,
        "run_generation": bool(run_generation),
        "provided_generated_scene": str(generated_scene) if generated_scene else None,
        "generation_run": getattr(adapter, "last_run_metadata", None),
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
    object_plan: dict | None = None,
    generated_scene: str | Path | None = None,
    adapter_config: dict | None = None,
    run_generation: bool = False,
) -> dict:
    """Interface-only entry point for generators that expect natural language."""

    generation_input = build_direct_natural_language_generation_input(
        request_id=request_id,
        instruction=instruction,
        scene_type=scene_type,
        room=room,
        object_plan=object_plan,
    )
    return run_generate(
        generation_input=generation_input,
        adapter_name=adapter_name,
        out_dir=out_dir,
        generated_scene=generated_scene,
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
    parser.add_argument("--adapter", default="passthrough")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generated-scene", default=None)
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--evaluation-report", default=None, help="Optional evaluate.py output to attach as self-reflection feedback.")
    parser.add_argument("--previous-generated-scene", default=None, help="Optional previous generated_scene.json to attach with self-reflection feedback.")
    parser.add_argument("--iteration", type=int, default=None, help="Self-reflection iteration index for adapter metadata and input.")
    args = parser.parse_args()

    generation_input = read_json(_path_arg(args.generation_input))
    adapter_config = read_json(_path_arg(args.adapter_config)) if args.adapter_config else None
    if adapter_config is not None and not isinstance(adapter_config, dict):
        parser.error("--adapter-config must point to a JSON object.")
    result = run_generate(
        generation_input=generation_input,
        adapter_name=args.adapter,
        out_dir=_path_arg(args.out_dir),
        generated_scene=_path_arg(args.generated_scene) if args.generated_scene else None,
        adapter_config=adapter_config,
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
