# Scene Harness Architecture

This repo is a scene construction/evaluation harness. It defines stable artifacts and adapter boundaries around scene generation methods; it is not a full generation framework and does not claim to be a final benchmark.

## Flow

```text
NL / request
-> object_plan
-> asset_selection
-> adapter_in
-> generator
-> adapter_out
-> generated_scene
-> evaluator
```

Expanded artifact flow:

```text
scene_request.json
-> object_plan.json
-> asset_selection.json
-> generation_input.json
-> method-specific input
-> method-specific output
-> generated_scene.json
-> evaluation_report.json
```

## Canonical Boundary

`generated_scene.json` is the evaluator boundary. `evaluate.py` consumes this canonical scene and does not need to know how the scene was generated.

Generation method authors should integrate through two adapters:

- input adapter: `generation_input.json -> method-specific input`
- output adapter: `method-specific output -> generated_scene.json`

This keeps generator-specific formats away from evaluator code.

## Before Generation

Inputs before generation do not include deterministic object poses. They contain:

- natural-language instruction
- room boundary and height
- object requirements
- selected real assets
- optional soft placement intents

Therefore this is not a reconstruction-error benchmark. Generated scenes are not expected to match one ground-truth pose. Evaluation focuses on validity, constraint satisfaction, and plausibility.

## Evaluation

Current evaluator families:

- `generic_validity_v0`: deterministic scene validity checks over final asset instances
- `OOR v0`: deterministic object-object relationship checks
- `OAR v0`: deterministic object-architecture relationship checks
- VLM/semantic judge paths remain separate and optional

The deterministic v0 evaluators use real asset metadata and bbox/OBB proxy geometry. Mesh and point cloud files may be preserved as references, but v0 evaluators do not load mesh-level or point-cloud-level geometry.

## Asset Grounding

`asset_selection.json` resolves object requirements into real asset instances. `jid` should match `asset_info.csv name_en`, not the CSV row id. Asset references may include:

```text
mesh_uri
pointcloud_uri
metadata_uri
```

These are references only. The harness uses metadata such as `transformed_size` or CSV `bbx` to build OBB proxies.

## Adapters

Built-in adapters:

- `passthrough`: use when a canonical `generated_scene.json` already exists
- `manual`: use for smoke tests or externally produced raw outputs

`generate.py` is a dispatcher. It prepares adapter input, optionally accepts an externally generated scene, and writes workflow status. It does not implement model-specific generation.

`evaluate.py` is the canonical evaluator entry point. It loads `generated_scene.json`, optionally enriches assets, and runs selected evaluators.
