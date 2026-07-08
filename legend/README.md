# Legend Compatibility

This folder collects old layout/bbox-facing entry points while the main project
moves to scene/assets as the canonical contract.

- `legend/schemas/legend_layout.schema.json` keeps the old layout schema.
- `legend/scripts/legend_*.py` names the old generation benchmark scripts
  explicitly as legend compatibility entry points.
- `benchmark.legend.*` exposes Python wrappers for old workflow, judge, and
  layout-evaluation calls.

New code should prefer `schemas/scene.schema.json`,
`benchmark.workflow.evaluate.evaluate_scene`, and
`benchmark.workflow.generation.generate_scene`.
