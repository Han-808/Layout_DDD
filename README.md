# 3D Layout Benchmark

Minimal, extensible benchmarking framework for explicit 3D scene layout generation by VLMs/LLMs.

This is not a 3D asset generation project. Model-facing outputs are plain JSON layouts with explicit bounding boxes. Scoring is deterministic Python logic.

## What Is Evaluated

- Schema validity: JSON structure, required fields, object IDs, numeric bbox fields, units, coordinate-system metadata.
- Physical validity: room containment, height bounds, object-object collisions, floor/support consistency.
- Spatial-relation validity: near, far, facing, against_wall, on_top_of, left_of, right_of, in_front_of, behind.
- Optional repair: the same target model can be asked to repair a layout from deterministic `feedback.json`.

## Representation

Layouts use meters and bbox objects:

```json
{
  "scene_id": "bedroom_001",
  "unit": "meter",
  "objects": [
    {
      "object_id": "bed_1",
      "category": "bed",
      "center": [2.5, 1.0, 0.3],
      "size": [2.0, 1.6, 0.6],
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
py scripts/run_single_case.py --case data/benchmark_cases/bm_instance_001.json --model mock --max_repair_iterations 0 --out outputs/debug_case
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

The `ollama` and `vllm` entries in `configs/model_config.yaml` use OpenAI-compatible local HTTP endpoints. Start the model server separately, or update `endpoint` and `model` for your local setup.

Validate a case:

```bash
py scripts/validate_case.py --case data/benchmark_cases/bm_instance_001.json
```

Run tests:

```bash
py -m pytest
```

## Design Notes

- LayoutGPT motivates structured, JSON-first output and few-shot prompt formatting.
- LayoutVLM motivates explicit layouts, spatial relations, and physical plausibility checks.
- Holodeck motivates separating LLM-authored constraints from deterministic solving/checking.
- Direct Numerical Layout Generation motivates evaluator-feedback-to-layout-update loops.
- Scenethesis motivates separating collision, physical plausibility, and refinement stages.
- PhyScene motivates physical categories such as collision, boundary, support, and reachability-style checks.

The current scaffold implements deterministic checks only. `visual_judge/` is a placeholder for a future auxiliary VLM judge and is not part of primary scoring.
