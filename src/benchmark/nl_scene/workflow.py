from __future__ import annotations

CURRENT_INPUT_CHAIN = "natural_language"
LEGEND_INPUT_CHAIN = False

from pathlib import Path
from typing import Any

from benchmark.nl_scene.asset_retrieval import retrieve_assets_for_scene_spec
from benchmark.nl_scene.converter import convert_nl_to_scene_spec
from benchmark.nl_scene.dummy_evaluator import evaluate_scene
from benchmark.utils.io import read_json, write_json


def run_nl_scene_workflow(
    *,
    instruction: str,
    out_dir: str | Path,
    asset_index_path: str,
    scene_type: str | None = None,
    room: dict | None = None,
    retrieval_k: int = 1,
    retriever_module_path: str | None = None,
    use_vlm_selector: bool = True,
    model_config: dict | None = None,
    generated_scene_path: str | Path | None = None,
    seed: int = 0,
) -> dict:
    """Run the current NL input workflow through retrieval and optional dummy evaluation."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_record = {
        "instruction": instruction,
        "scene_type": scene_type,
        "room": room,
        "asset_index_path": str(asset_index_path),
        "retrieval_k": int(retrieval_k),
        "generated_scene_path": str(generated_scene_path) if generated_scene_path else None,
    }
    artifacts: dict[str, str] = {}
    artifacts["input"] = str(write_json(output_dir / "input.json", input_record))
    scene_spec = convert_nl_to_scene_spec(instruction, scene_type=scene_type, room=room, model_config=model_config)
    artifacts["scene_spec"] = str(write_json(output_dir / "scene_spec.json", scene_spec))
    retrieval = retrieve_assets_for_scene_spec(
        scene_spec,
        asset_index_path=asset_index_path,
        retrieval_k=retrieval_k,
        retriever_module_path=retriever_module_path,
        use_vlm_selector=use_vlm_selector,
        model_config=model_config,
    )
    artifacts["asset_retrieval"] = str(write_json(output_dir / "asset_retrieval.json", retrieval))
    generation_input = build_generation_input(instruction=instruction, scene_spec=scene_spec, retrieval=retrieval, room=room)
    artifacts["generation_input"] = str(write_json(output_dir / "generation_input.json", generation_input))
    if generated_scene_path:
        generated_scene = read_json(generated_scene_path)
        report = evaluate_scene(generated_scene, instruction=instruction, seed=seed, dummy=True)
        artifacts["evaluation_report"] = str(write_json(output_dir / "evaluation_report.json", report))
        status = {"status": "evaluated", "reason": "generated scene was provided", "evaluation_report": "evaluation_report.json"}
    else:
        status = {
            "status": "generation_skipped",
            "reason": "generation stage is not implemented yet",
            "next_expected_input": "generated_scene.json",
        }
    artifacts["workflow_status"] = str(write_json(output_dir / "workflow_status.json", status))
    return {"artifacts": artifacts, "scene_spec": scene_spec, "asset_retrieval": retrieval, "workflow_status": status}


def build_generation_input(*, instruction: str, scene_spec: dict, retrieval: dict, room: dict | None = None) -> dict:
    selected_assets = []
    for item in retrieval.get("objects", []) if isinstance(retrieval, dict) else []:
        if isinstance(item, dict):
            selected_assets.append(
                {
                    "object_spec": item.get("object_spec"),
                    "selected_asset": item.get("selected_asset"),
                    "selection_reason": item.get("selection_reason"),
                }
            )
    return {"original_instruction": instruction, "scene_spec": scene_spec, "selected_assets": selected_assets, "room": room}


def generate_scene_placeholder(*_: Any, run_generation: bool = False, **__: Any) -> None:
    if run_generation:
        raise NotImplementedError("generation stage is not implemented yet")
    return None
