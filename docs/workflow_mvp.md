# Natural-Language Scene Workflow MVP

This is the current natural-language input path. It is intentionally artifact-first and still incomplete, but it is no longer an HSSD/BM-instance workflow. HSSD/BM-instance inputs are retained only as legend compatibility adapters.

## Current Flow

```text
natural language scene/request
-> VLM/LLM converter
-> scene_spec.json
   - object retrieval specs
   - optional weak/pass-through relation hints
   - no placement coordinates
   - no exact asset ids
-> asset retriever, called by the workflow once per object spec
-> asset_retrieval.json
-> generation_input.json
-> generation skipped unless a generated scene is supplied
-> optional dummy smoke evaluation for supplied generated scene JSON
```

Important boundary: the converter itself does not call the retriever today. The workflow calls `retrieve_assets_for_scene_spec(...)` after `convert_nl_to_scene_spec(...)` returns a scene spec.

## Converter Status

The converter currently maps natural language to a structured scene spec for retrieval. Objects contain:

```text
id
role
category
description
count
estimated_size optional
```

Top-level scene spec contains:

```text
scene_type
scene_description
objects
global_constraints
relations
```

`relations` are currently weakly supported: if the model returns relation dictionaries, the converter preserves them, but it does not yet normalize them into evaluator-ready OOR/OAR relation schemas.

So the current status is:

```text
NL -> object retrieval spec: implemented
NL -> asset retrieval through workflow: implemented
NL -> evaluator-ready relationships: not fully implemented
NL -> final generated real-asset scene: not implemented in this MVP path
```

## Required Converter Upgrade

The converter needs a relationship-aware upgrade. The target shape is:

```json
{
  "scene_type": "living room",
  "scene_description": "...",
  "objects": [
    {
      "id": "chair_1",
      "role": "main seating",
      "category": "chair",
      "description": "comfortable reading chair",
      "estimated_size": [0.8, 0.8, 1.0],
      "count": 1
    }
  ],
  "relations": [
    {
      "family": "oor",
      "type": "near",
      "subject": "chair_1",
      "object": "table_1"
    },
    {
      "family": "oar",
      "type": "against_wall",
      "subject": "bookshelf_1",
      "wall": "north"
    }
  ],
  "global_constraints": ["walkable", "cozy"]
}
```

The upgraded converter should normalize natural-language relationship phrases into evaluator families:

```text
OOR: near, left, right, in_front, behind, above, below, aligned_with, contact, face_to, within, out_of
OAR: on_floor, against_wall, near_wall, below_wall, at_corner
Generic validity: no semantic relation input; it evaluates final geometry/assets only
```

These relations should then be carried forward into generation input and final scene artifacts so OOR/OAR can evaluate them deterministically after placement.

## Retriever Status

The retriever attaches asset candidates to each requested object. When `retrieval_k=1`, the top candidate is selected automatically. When `retrieval_k>1`, candidates are preserved and an optional VLM selector can choose a final asset.

The retriever is object-level, not scene-level:

```text
one object spec -> asset candidates
```

The workflow repeats this for every object in the converted scene spec.

## Generation And Evaluation Status

Scene generation and placement are still not implemented in this NL MVP path. If no generated scene is provided, the workflow writes `generation_input.json` and exits with `workflow_status.json` set to `generation_skipped`.

Evaluation status has changed since the original MVP:

```text
generic_validity_v0: deterministic scene validity checks over final asset instances
OOR v0: deterministic object-object relationship checks
OAR v0: deterministic object-architecture relationship checks
VLM-as-judge: separate visual/semantic judge path
dummy_v0: legacy smoke-test evaluator only
```

The dummy evaluator should not be treated as the real evaluation path. It remains useful only for quick artifact plumbing tests when a generated scene JSON is supplied.

## Example Workflow

```bash
python scripts/run_nl_scene_workflow.py \
  --instruction "Create a cozy living room for reading and casual conversation. Include seating, a rug, coffee table, lighting, and decor. Keep it walkable." \
  --scene-type "living room" \
  --asset-index-path outputs/asset_index_imaginarium \
  --retrieval-k 1 \
  --out-dir outputs/mvp_demo
```

This writes:

```text
input.json
scene_spec.json
asset_retrieval.json
generation_input.json
workflow_status.json
```

If you already have a generated scene JSON:

```bash
python scripts/run_nl_scene_workflow.py \
  --instruction "Create a cozy living room for reading and casual conversation." \
  --asset-index-path outputs/asset_index_imaginarium \
  --generated-scene outputs/some_generated_scene.json \
  --out-dir outputs/mvp_demo_eval \
  --seed 0
```

## Direct Evaluators

Generic deterministic validity:

```bash
python scripts/evaluate_generic_validity.py \
  --scene outputs/generated_scene.json \
  --out outputs/generic_validity_report.json
```

OOR deterministic relationships:

```bash
python scripts/evaluate_oor.py \
  --scene outputs/generated_scene.json \
  --out outputs/oor_report.json
```

OAR deterministic relationships:

```bash
python scripts/evaluate_oar.py \
  --scene outputs/generated_scene.json \
  --out outputs/oar_report.json
```

Dummy smoke evaluation, kept only for compatibility:

```bash
python scripts/evaluate_scene.py \
  --scene outputs/some_generated_scene.json \
  --instruction "Create a cozy living room..." \
  --out outputs/evaluation_report.json \
  --seed 0 \
  --dummy
```

Omit `--dummy` to use the existing full scene evaluation path.
