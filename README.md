# 3D Layout Benchmark

Minimal, extensible workflow for explicit 3D scene layout generation by VLMs/LLMs, with VLM-as-judge evaluation over rendered bbox views.

This is not a 3D asset generation project. Model-facing outputs are plain JSON layouts with explicit bounding boxes. The main evaluator renders view evidence and asks a VLM judge to score the generated scene.

## What Is Evaluated

- Non-blocking schema/layout sanity: parseable layouts are judged by the VLM; schema and bbox issues become judge-facing flags.
- VLM-as-judge scene quality: Qwen3-VL/OpenAI-compatible judge scores rendered global and object-group views.
- Auxiliary physical flags: room boundary, below floor, above wall height, serious collision above 50% overlap.
- VLM relation/attachment judgement: explicit visible relations and attachments are included in the judge context.
- Optional repair: the same target model can be asked to repair a layout from deterministic `feedback.json`.

## Representation

Layouts use meters and bbox objects:

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

Validity fields are never stored in `layout.json`; they belong only in `evaluation_report.json`.

## Install

```bash
py -m pip install -e .[dev]
```

On Unix-like systems, use `python` instead of `py`.

## Run

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

The `qwen3vl_sglang`, `qwen3vl_sglang_32b`, `ollama`, and `vllm` entries in `configs/model_config.yaml` are server/API profiles. Start the model server separately, or add/update a model entry for your setup. Case-specific choices such as case path, output directory, and larger HSSD token budgets belong in `configs/experiment_config.yaml`. The default VLM judge reuses the same model endpoint as the layout generator; pass `--judge_model` only when you intentionally want a separate configured judge model. Use `scripts/check_model_endpoint.py` for temporary endpoint/model-id smoke checks instead of adding server debug parameters to the main pipeline commands.

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

The current main pipeline uses `vlm_as_judge_v1`: global top view plus object-group `xy/yz/xz` views are sent to a VLM judge. Deterministic schema, renderability, and physical logic is auxiliary evidence; only fully unparseable model output skips VLM judging.
