# 3D Layout Benchmark

Evaluation-first benchmark for explicit 3D scene/layout validity. The core module evaluates canonical scene JSON with a VLM-as-judge; layout generation is an optional harness that produces a candidate scene and then calls the same evaluator.

This is not a 3D asset generation project. Inputs and outputs are JSON scene/layout records with asset metadata and optional bounding boxes. The bbox is a geometry proxy for placement evidence, while `asset_ref` is future-facing metadata for catalogs, meshes, or templates. The evaluator can run from structured JSON alone or from JSON plus rendered bbox views.

## What Is Evaluated

- Canonical scene JSON: `schemas/scene.schema.json` describes asset-aware evaluation inputs; legacy bbox layouts are adapted into this scene form.
- Non-blocking schema/layout sanity: parseable scenes/layouts are judged by the VLM; schema and bbox issues become judge-facing flags.
- VLM-as-judge scene quality: Qwen3-VL/OpenAI-compatible judge scores structured evidence, optionally with rendered global and object-group views.
- Auxiliary physical flags: room boundary, below floor, above wall height, serious collision above 50% overlap.
- VLM relation/attachment judgement: explicit visible relations and attachments are included in the judge context.
- Advisory feedback: `feedback.json` reports issues, physical evidence, VLM judge feedback, and suggested actions. The same data can be used for optional repair loops.

## Representation

Canonical evaluation scenes use meters, assets, optional `asset_ref` metadata, and optional bbox placement:

```json
{
  "scene_id": "candidate_scene",
  "unit": "meter",
  "assets": [
    {
      "asset_id": "chair_1",
      "category": "chair",
      "asset_ref": {"source": "hssd-hab", "template_id": "chairs/example"},
      "bbox": {"center": [1.0, 1.5, 0.45], "size": [0.6, 0.6, 0.9], "yaw": 180}
    }
  ]
}
```

Legacy generated layouts are still supported and are adapted into scene assets:

```json
{
  "scene_id": "102344115_structured_basic",
  "unit": "meter",
  "objects": [
    {
      "object_id": "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
      "category": "20b73dd1f91dd128fb928fb7a032af2a47e79882",
      "center": [-1.76, -2.66, 0.5],
      "size": [1.0, 1.0, 1.0],
      "yaw": 0,
      "support_parent": "floor"
    }
  ]
}
```

Validity fields are never stored in scene/layout input JSON; they belong only in `evaluation_report.json` and `feedback.json`.

## Architecture

The primary path is:

```text
scene / candidate_scene -> evaluation core -> VLM judge -> metrics -> feedback
```

Generation mode is a compatibility harness:

```text
input case -> model generation -> candidate layout/scene -> evaluation core
```

Programmatic generation-mode runs use an agent-style runner rather than a LangGraph-defined fixed state machine:

```python
from benchmark.workflow import BenchmarkAgent

state = BenchmarkAgent().run(initial_state)
```

`BenchmarkAgent` accepts a replaceable policy and callbacks for observing each action. The legacy `run_workflow(state)` and `build_graph().invoke(state)` entry points are still present as compatibility wrappers around the same runner.

The Three.js viewer is visualization-only. Blender, Habitat rendering, photorealistic rendering, and mesh-based evaluation are not part of the current evaluator. Missing meshes or textures are not validity failures. Physical flags such as boundary, height, support, and collision diagnostics are evidence for the VLM-as-judge, not final deterministic verdicts; relation-dependent overlaps such as support, containment, attachment, or contact are judged semantically.

## Install

```bash
py -m pip install -e .[dev]
```

On Unix-like systems, use `python` instead of `py`.

## Run

Direct scene evaluation:

```bash
py scripts/evaluate_scene.py --scene data/scenes/candidate_scene.json --out outputs/eval_scene --judge-model qwen3vl_sglang_32b --vlm-judge-input-mode json_only
```

Use `--vlm-judge-input-mode json_only` for structured JSON evidence only. Use `--vlm-judge-input-mode json_plus_render` or `--json-plus-render` when bbox render evidence should be generated and sent to the judge.

Single case:

```bash
py scripts/run_single_case.py --case data/benchmark_cases/hssd_small_room_full/102344115_structured_basic.json --model mock --max_repair_iterations 0 --out outputs/hssd_small_room_debug_case
```

Single case from an experiment profile:

```bash
py scripts/run_single_case.py --experiment hssd_small_room_qwen3vl32b_local --out outputs/hssd_small_room_vlm_judge_smoke --serve --port 8080
```

Benchmark folder:

```bash
py scripts/run_benchmark.py --cases data/benchmark_cases --model mock --max_repair_iterations 0 --out outputs/benchmark_run
```

HSSD-HAB with a local open-source model:

```bash
py scripts/prepare_hssd_hab.py --hssd-root data/external/hssd-hab
py scripts/convert_hssd_hab.py --hssd-root data/external/hssd-hab --out-dir data/benchmark_cases/hssd --limit 1 --levels structured_basic structured_relation
py scripts/run_single_case.py --case data/benchmark_cases/hssd/<case_id>_structured_basic.json --model ollama --max_repair_iterations 0 --out outputs/hssd_local_model
```

The `qwen3vl_sglang`, `qwen3vl_sglang_32b`, `ollama`, and `vllm` entries in `configs/model_config.yaml` are server/API profiles. Start the model server separately, or add/update a model entry for your setup. Case-specific choices such as case path, output directory, and larger HSSD token budgets belong in `configs/experiment_config.yaml`. The default VLM judge reuses the same model endpoint as the layout generator in generation mode; pass `--judge_model` or direct-eval `--judge-model` when you intentionally want a separate configured judge model. Use `scripts/check_model_endpoint.py` for temporary endpoint/model-id smoke checks instead of adding server debug parameters to the main pipeline commands.

Validate a case:

```bash
py scripts/validate_case.py --case data/benchmark_cases/hssd_small_room_full/102344115_structured_basic.json
```

Run tests:

```bash
py -m pytest
```

## Design Notes

- LayoutGPT motivates structured, JSON-first output and few-shot prompt formatting.
- LayoutVLM motivates explicit layouts, spatial relations, rendered views, and VLM-as-judge evaluation.
- Holodeck motivates separating LLM-authored constraints from deterministic solving/checking.
- Direct Numerical Layout Generation motivates evaluator-feedback-to-layout-update loops.
- Scenethesis motivates separating collision, physical plausibility, and refinement stages.
- PhyScene motivates physical diagnostic categories such as collision, boundary, support, and reachability-style checks.

The current evaluator uses `vlm_as_judge_v1`. In `json_only` mode, the judge receives scene JSON, relation/attachment specs, physical flags, grouping evidence, and missing-bbox evidence without images. In `json_plus_render` mode, global top views and object-group `xy/yz/xz` bbox views are also sent. Deterministic schema, renderability, and physical logic is auxiliary evidence; only fully unparseable model output skips VLM judging.

Pipeline outputs include `pipeline_mode` and `generation_used`. Direct scene evaluation writes `pipeline_mode: "evaluation"` and `generation_used: false`; generation runs write `pipeline_mode: "generation"` and `generation_used: true`.
