# Legend Compatibility

This folder collects old layout/bbox-facing entry points while the main project
moves to scene/assets as the canonical contract.

- `legend/schemas/legend_layout.schema.json` keeps the old layout schema.
- `scripts/legend/legend_*.py` names the old generation benchmark scripts
  explicitly as legend compatibility entry points.
- `benchmark.legend.*` exposes Python wrappers for old workflow, judge, and
  layout-evaluation calls.

Legend modules are retained in the source tree for archaeology and migration,
but are excluded from the installed package. New code should use
`benchmark.api.generation`, `benchmark.api.evaluation`, and
`benchmark.api.submission` with `schemas/scene.schema.json`.

Known frozen limitation: the historical default root calculation in
`benchmark.legend.data.local_assets` and `benchmark.legend.data.local_scenes`
resolves below `src/` rather than at the repository root. The replay code is
hash-pinned, so this defect is intentionally preserved. Archaeology or replay
callers must pass the repository root explicitly; these modules are not a
supported current-runtime compatibility surface.
