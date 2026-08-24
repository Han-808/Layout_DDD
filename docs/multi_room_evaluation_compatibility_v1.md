# Multi-room evaluation compatibility v1

`benchmark.multi_room_evaluation` is an additive, read-only bridge between the
multi-room generation mode and the existing scene-level evaluator. It does not
change evaluator prompts, camera policy, metric weights, deduction multiplier,
publishability, selection, score aggregation, or the `run_evaluate` call.

## Scientific boundary

Each successfully generated room projection becomes one ordinary evaluator
case. The existing campaign therefore evaluates and aggregates rooms exactly as
it already evaluates ordinary independent cases. Failed and missing generation
rooms remain in the dataset ledger and are never represented as successful
cases. The following scopes remain unsupported:

- `cross_room_collision`
- `cross_room_functionality`
- `global_architecture_scoring`
- `multi_room_overall_score`

Official materialization requires one model and no failed or missing expected
rooms. `--allow-incomplete` is diagnostic: it materializes succeeded rooms,
sets `all_materialized_cases_ready=true`, but also sets
`all_expected_rooms_ready=false`, `diagnostic_incomplete=true`, and
`official_full_model_score_eligible=false`.
Diagnostic datasets are intentionally rejected by the campaign-config builder,
because the unchanged finalizer labels its selected-case aggregate as official.

## Trust and discovery

Discovery follows `collection_manifest.json` to each model's
`selection_manifest.json`. A selected layout is the one permitted parent
symlink. Its target must equal the selection manifest's declared source path.
The collection manifest pins each selection-manifest hash; every non-missing
selection row pins both `room_evaluation_index.json` and
`assembly_manifest.json`. The assembly manifest then pins the floor plan, all
room results, compiled architecture, global scene, index, and room projections.
The reader then loads the nested `room_evaluation_index.json`, calls the
generation-owned `validate_evaluation_index` with the resolved layout root, and
independently verifies:

- collection, selection, layout and room identities and exact counts;
- floor-plan canonical hash and compiled-architecture hash;
- source room-result path and hash;
- all six evaluator source inputs and their hashes;
- exact, non-symlink evaluator artifact-tree coverage;
- safe paths, unique room identities, and unique flattened case IDs.

Missing layouts have no index. Their rooms remain explicit from the selection
ledger with `room_id=null`; the compatibility layer never guesses an identity
that the source did not record.

## Official render profile

`--require-complete` also requires the pinned
`room_evaluation_official_render_v1` profile. The CLI defaults are the same as
the existing formal evaluation datasets:

- `BLENDER_EEVEE_NEXT`, CPU;
- 768 × 768;
- 16 configured Cycles samples and denoising disabled (dormant under EEVEE);
- asset meshes required;
- collision limits: 400,000 vertices/object, 550,000 faces/object,
  2,000,000 total vertices, and 2,500,000 total faces.

Changing any of these settings while using `--require-complete` fails before
rendering. Alternative renderer settings require `--allow-incomplete` and
produce a diagnostic, campaign-ineligible dataset even when all source rooms
are present.

## Materialization and resume

The only rendering implementation is `benchmark.rendering.BlenderRenderer`.
Each ready case contains the historical evaluator layout:

```text
<dataset>/<case-id>/
  case_manifest.json
  annotation.json
  scene/canonical_scene.json
  prepared/evaluation.blend
  evidence/standardized_perspective.png
  evidence/standardized_top.png
  evidence/standardized_identity_map.png
  evidence/prepared_render_manifest.json
  evidence/collision_geometry_manifest.json
  evidence/collision_geometry/*
  provenance/source_inputs/{canonical_scene,scene_request,object_plan,
    asset_selection,generation_input,architecture_contract}.json
  provenance/source_inputs/manifest.json
  provenance/materialization_receipt.json
```

Generation instructions are retained only under `provenance/source_inputs`.
The evaluator continues to construct its existing promptless request with an
empty instruction. Every annotation is non-authoritative: `reviewed=false`, all
five L3 metrics use `anomaly=false, unclear=true`, and the case is excluded from
human-accuracy comparison.

Writes occur beneath `.<output-name>.building`. A completed case is skipped
only after its source identity, receipt, complete artifact inventory, hashes,
collision manifest, runtime source closure, Blender executable, catalog, and
all referenced asset bytes are revalidated. Output/building roots must be
disjoint from the collection, selected generation roots, and asset root. A partial case directory is never
discoverable as ready. Final output is published by one directory rename; an
existing final dataset is never overwritten.

## Build one model dataset

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python \
  scripts/prepare_multi_room_evaluation_dataset.py \
  --collection-root scene10_all_models_v1 \
  --model gpt-5.6-sol \
  --output-root Support/datasets/multi_room/gpt-5.6-sol-v1 \
  --blender-bin /Applications/Blender.app/Contents/MacOS/Blender \
  --asset-root Support/Assets/imaginarium_assets \
  --require-complete
```

Multiple `--model` values create separate datasets beneath the supplied output
root. They are never mixed into one evaluator aggregation.

To write a campaign config at the same time, add a reviewed existing template
and disjoint output roots:

```bash
  --campaign-template configs/evaluation/campaigns/glm53_api1_full10_v1.json \
  --campaign-config-out configs/evaluation/campaigns/mr_gpt56sol_v1.json \
  --campaign-id mr-gpt56sol-room-evaluation-v1 \
  --attempt-parent Support/artifacts/outputs/multi_room_eval/gpt56sol_attempts \
  --final-selection-root Support/artifacts/outputs/multi_room_eval/gpt56sol_final
```

The builder copies the template's Judge profile, kernel, renderer/evaluator
configuration, retry policy, deduction multiplier, and first-publishable
selection policy unchanged. It replaces only campaign/model identity, dataset
identity and case lists, and output roots.

## Run the existing evaluation campaign

Static check (no binding or network):

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmark.evaluation_campaign check \
  --config configs/evaluation/campaigns/mr_gpt56sol_v1.json
```

Run with the existing private binding contract:

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmark.evaluation_campaign run \
  --config configs/evaluation/campaigns/mr_gpt56sol_v1.json \
  --bindings configs/evaluation/campaigns/evaluation_bindings.local.json
```

No generation output, retry root, evaluator module, prompt, camera component, or
scoring implementation is modified by these commands.
