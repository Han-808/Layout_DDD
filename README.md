# 3D Layout Benchmark

Evaluation-first benchmark for explicit 3D scene asset validity. The core module evaluates canonical scene JSON with a VLM-as-judge; layout generation is an optional harness that produces a candidate scene and then calls the same evaluator.

This is not a 3D asset generation project. Inputs and outputs are JSON scene records with assets, placement, dimensions, local `scene_ref` metadata pointing at `Scenes/`, and `asset_ref` metadata pointing at local repo assets under `Assets/imaginarium_assets`. Old layout/bbox-facing entry points live under `legend/` and `benchmark.legend.*`. The evaluator can run from structured JSON alone or from JSON plus rendered geometry-proxy views.

## What Is Evaluated

- Canonical scene JSON: `schemas/scene.schema.json` describes asset-aware evaluation inputs with `placement` and `dimensions`.
- Non-blocking schema/scene sanity: parseable scenes are judged by the VLM; schema and geometry issues become judge-facing flags.
- VLM-as-judge scene quality: Qwen3-VL/OpenAI-compatible judge scores structured evidence, optionally with rendered global and object-group views.
- Auxiliary physical flags: room boundary, below floor, above wall height, serious collision above 50% overlap.
- VLM relation/attachment judgement: explicit visible relations and attachments are included in the judge context.
- Advisory feedback: `feedback.json` reports issues, physical evidence, VLM judge feedback, and suggested actions. The same data can be used for optional repair loops.

## Representation

Canonical evaluation scenes use meters, assets, local `asset_ref` metadata, placement, and dimensions:

```json
{
  "scene_id": "candidate_scene",
  "unit": "meter",
  "assets": [
    {
      "asset_id": "chair_1",
      "category": "chair",
      "asset_ref": {
        "source": "local_repo",
        "collection": "imaginarium_assets",
        "asset_id": "0_SM_Chair_1",
        "repo_path": "Assets/imaginarium_assets/0_SM_Chair_1",
        "mesh_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1.fbx",
        "pointcloud_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1.ply",
        "metadata_path": "Assets/imaginarium_assets/0_SM_Chair_1/0_SM_Chair_1_metadata.json"
      },
      "placement": {"position": [1.0, 1.5, 0.45], "yaw_degrees": 180},
      "dimensions": [0.646512, 0.637464, 0.850842]
    }
  ]
}
```

When `asset_ref.source` is `local_repo`, the loader resolves `asset_ref.asset_id` against `Assets/imaginarium_assets/<asset_id>/` and fills repo-relative mesh, pointcloud, metadata, category, and dimensions when available.

Local scenes under `Scenes/` are also first-class inputs. Raw scene JSON files with `objects` are normalized into assets, preserving `scene_id`, `scene_type`, room boundary, wall height, `jid`, placement, dimensions, and rotation. A compact reference can load the same local JSON:

```json
{
  "scene_id": "scene_000000_03",
  "scene_ref": {
    "source": "local_repo",
    "scene_id": "scene_000000_03"
  }
}
```

Legend compatibility layouts are still supported through `legend/` and are adapted into scene assets:

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

Validity fields are never stored in scene input JSON; they belong only in `evaluation_report.json` and `feedback.json`.

## Architecture

The primary path is:

```text
scene / candidate_scene -> evaluation core -> VLM judge -> metrics -> feedback
```

Generation mode is a compatibility harness:

```text
input case -> model generation -> candidate scene -> evaluation core
```

Programmatic generation-mode runs use an agent-style runner rather than a LangGraph-defined fixed state machine:

```python
from benchmark.workflow import BenchmarkAgent

state = BenchmarkAgent().run(initial_state)
```

`BenchmarkAgent` accepts a replaceable policy and callbacks for observing each action. The legend `run_workflow(state)` and `build_graph().invoke(state)` entry points are still present as compatibility wrappers around the same runner.

The Three.js viewer is visualization-only. Blender, Habitat rendering, photorealistic rendering, and mesh-based evaluation are not part of the current evaluator. Missing meshes or textures are not validity failures. Physical flags such as boundary, height, support, and collision diagnostics are evidence for the VLM-as-judge, not final deterministic verdicts; relation-dependent overlaps such as support, containment, attachment, or contact are judged semantically.

## Install

```bash
py -m pip install -e .[dev]
```

On Unix-like systems, use `python` instead of `py`.

## Run

Direct scene evaluation:

```bash
py scripts/evaluate_scene.py --scene Scenes/converted_scenes/scene_000000_03.json --out outputs/eval_scene --judge-model qwen3vl_sglang_32b --vlm-judge-input-mode json_only
```

Use `--vlm-judge-input-mode json_only` for structured JSON evidence only. Use `--vlm-judge-input-mode json_plus_render` or `--json-plus-render` when geometry-proxy render evidence should be generated and sent to the judge.

Direct scene generation:

```bash
py scripts/generate_scene.py --case data/benchmark_cases/hssd_small_room_full/102344115_structured_basic.json --model mock --out outputs/generated_scene
```

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

The current evaluator uses `vlm_as_judge_v1`. In `json_only` mode, the judge receives scene JSON, relation/attachment specs, physical flags, grouping evidence, and missing-geometry evidence without images. In `json_plus_render` mode, global top views and object-group `xy/yz/xz` geometry-proxy views are also sent. Deterministic schema, renderability, and physical logic is auxiliary evidence; only fully unparseable model output skips VLM judging.

Pipeline outputs include `pipeline_mode` and `generation_used`. Direct scene evaluation writes `pipeline_mode: "evaluation"` and `generation_used: false`; generation runs write `pipeline_mode: "generation"` and `generation_used: true`.
