# Layout_DDD

Layout_DDD generates and evaluates 3D scene layouts. Core APIs live in
`src/benchmark`; workflows, contracts and operator guidance are in `docs`.

## Latest-designated floor-plan evaluators

Use the mode-aware repository entrypoint to select and verify the evaluator
that is actually designated for each floor-plan mode:

```bash
python3 scripts/run_floorplan_evaluator.py --mode single_room
python3 scripts/run_floorplan_evaluator.py --mode multi_room
python3 scripts/run_floorplan_evaluator.py --mode non_rectangular_multi_room
```

These commands only describe and verify. To inspect the executable Single-room
runner's options without evaluating anything:

```bash
python3 scripts/run_floorplan_evaluator.py --mode single_room --run -- --help
```

| Mode | Source in this repository |
|---|---|
| Single-room / Open-space | [Exact published evaluator](evaluator_snapshots/single_room_sceneweaver_20260909_v1/README.md): Collision v4, Support v12, OOB v2, L3 v8, prompt v31 |
| Nonrect | Root package, matching the `3186983` core, plus execution v4; the original campaign recipe still needs private/local sealed releases |
| Rectangular Multi-room | Historical baseline record only; exact historical source is not recovered and is never silently substituted |

The [baseline registry](configs/runners/floorplan_evaluator_baselines_v1.json)
defines the current mapping. Read the [publication and scoring source guide](docs/evaluator_publication_20260911.md)
for source identity, executable examples and known limitations. Existing
low-level APIs and legacy entrypoints keep their compatibility behavior; they
do not automatically select the latest mode-specific snapshot.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest tests/test_floorplan_evaluator_publication.py
```

Real evaluation additionally requires the selected workflow's assets, prepared
inputs, Blender and private model credentials. Never commit credentials, raw API
exchanges or generated scene/evaluation artifacts. Publishing source does not
recompute historical scores or change any already running evaluation.
