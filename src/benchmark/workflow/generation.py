from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from benchmark.data.scene_adapters import layout_to_scene
from benchmark.models.base_model import ModelResponseError, build_generation_prompt
from benchmark.models.prompt_budget import PromptBudgetError
from benchmark.utils.io import write_json
from benchmark.workflow.agent import BenchmarkAgent
from benchmark.workflow.layout_normalization import enforce_layout_object_set
from benchmark.workflow.state import BenchmarkState


def generate_scene(
    case: dict,
    *,
    model: Any,
    scene_schema: dict | None = None,
    legend_layout_schema: dict | None = None,
    benchmark_config: dict | None = None,
    out_dir: str | Path | None = None,
) -> tuple[dict, dict]:
    """Generate a canonical scene from a benchmark case.

    Current model adapters still speak the old layout-generation protocol. This
    function keeps that protocol behind a scene-first API by converting the
    generated layout into an asset scene before returning it.
    """

    output_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="layout_ddd_generate_scene_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_schema = legend_layout_schema or scene_schema or {}
    metadata: dict[str, Any] = {
        "generation_api": "benchmark.workflow.generation.generate_scene",
        "generation_used": True,
        "pipeline_mode": "generation",
    }
    try:
        layout = model.generate_layout(case, layout_schema)
    except PromptBudgetError as exc:
        layout = _empty_legend_layout(case)
        metadata["generation_error"] = str(exc)
        metadata["prompt_budget_exceeded"] = True
        metadata["prompt_budget_error_stage"] = "generation"
    except ModelResponseError as exc:
        layout = _empty_legend_layout(case)
        metadata["generation_error"] = str(exc)

    if not metadata.get("generation_error"):
        layout, normalization = enforce_layout_object_set(layout, case, stage="generation")
        metadata["layout_normalization"] = normalization

    scene = layout_to_scene(layout, case)
    source = scene.get("source") if isinstance(scene.get("source"), dict) else {}
    source.update(
        {
            "generation_api": "benchmark.workflow.generation.generate_scene",
            "legend_layout_compat": True,
        }
    )
    scene["source"] = source

    generated_scene_path = write_json(output_dir / "generated_scene.json", scene)
    legend_layout_path = write_json(output_dir / "legend_generated_layout.json", layout)
    metadata.update(
        {
            "generated_scene_path": generated_scene_path.as_posix(),
            "legend_layout_path": legend_layout_path.as_posix(),
        }
    )
    request_metadata = getattr(model, "last_request_metadata", None)
    if isinstance(request_metadata, dict):
        metadata["request_metadata_path"] = write_json(output_dir / "generation_request_metadata.json", request_metadata).as_posix()
    prompt_text = getattr(model, "last_prompt_text", "") or build_generation_prompt(case, layout_schema)
    if prompt_text:
        prompt_path = output_dir / "generation_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        metadata["prompt_path"] = prompt_path.as_posix()
    raw_text = getattr(model, "last_response_text", "") or json.dumps(scene, ensure_ascii=False, indent=2)
    if raw_text:
        raw_path = output_dir / "generation_raw_response.txt"
        raw_path.write_text(raw_text, encoding="utf-8")
        metadata["raw_response_path"] = raw_path.as_posix()
    return scene, metadata


def run_generation_workflow(initial_state: BenchmarkState, *, agent: BenchmarkAgent | None = None) -> BenchmarkState:
    """Run the generation workflow through the agent-style API."""

    runner = agent or BenchmarkAgent()
    state = dict(initial_state)
    state.setdefault("pipeline_mode", "generation")
    state.setdefault("generation_used", True)
    return runner.run(state)


def _empty_legend_layout(case: dict) -> dict:
    scene_id = case.get("case_id") or case.get("task_id") or "scene"
    return {
        "scene_id": scene_id,
        "unit": "meter",
        "coordinate_system": {
            "origin": "case floor-plan coordinate frame",
            "x_axis": "floor-plan x coordinate",
            "y_axis": "floor-plan y/depth coordinate",
            "z_axis": "height",
            "rotation_unit": "degree",
        },
        "objects": [],
    }
