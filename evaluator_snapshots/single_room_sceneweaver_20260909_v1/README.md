# Single-room evaluator: executable source release

This directory publishes the **actual Single-room evaluator source** designated
by `single_room_sceneweaver_20260909_v1`, not a reconstruction from Nonrect.
It is executable from a repository checkout. Collision v4, Support v12, OOB v2,
L3 v8 and prompt v31 are retained byte-for-byte, including scoring, evidence,
fallback and incomplete-coverage behavior.

## Select and verify from the repository root

```bash
python3 scripts/run_floorplan_evaluator.py --mode single_room
python3 scripts/run_floorplan_evaluator.py --mode single_room --run -- --help
```

The first command only checks source integrity and describes the selected
baseline. The second checks integrity and displays the frozen runner's help;
neither starts evaluation. The selector starts the frozen entrypoint in a new
Python interpreter, so it cannot accidentally reuse already imported Nonrect
`benchmark` modules. The original `src/benchmark` and its existing entrypoints
remain unchanged for compatibility; use this selector when asking for the
registered latest-used floor-plan baseline.

An intentional **new** evaluation requires a prepared camera-cal dataset,
Blender, assets and private Judge configuration. For example, after installing
the repository dependencies into your own environment:

```bash
python3 scripts/run_floorplan_evaluator.py --mode single_room --run -- \
  --dataset-root /path/to/prepared-dataset \
  --output-root /path/to/new-evaluation-output \
  --grouping-config evaluator_snapshots/single_room_sceneweaver_20260909_v1/configs/grouping/vlm_visual_evidence_scope_v2.yaml \
  --functional-group-local-granularity per_check \
  --functional-group-local-evidence-policy shared_group_bank \
  --deduction-multiplier 2 \
  --max-workers 5
```

Configure the Judge through the runner's supported environment variables
(`JUDGE_ENDPOINT`, `JUDGE_MODEL`, `JUDGE_API_KEY_ENV`), never by committing secrets.
Execution options and input manifests must be recorded for each new run. This
command selects exact evaluator **code**, not a promise that arbitrary arguments
reproduce the historical dataset, execution policy or results. Do not point a
new launch at an ongoing or historical output directory.

## Scope and provenance

`source_manifest.json` records hashes for the original **860-file** frozen tree
(`893b3a5c38751a553f348ac2e12a0142121cf489858abb28a66152f33ed069dd`).
The published exact subset contains **484 files**: the whole `src/benchmark`
package, evaluator/grouping configurations, the reference proxy configuration,
and the camera-cal scene-level entrypoint. Its independently named subset hash
is `a9caf5cea2a0201a6eac69a84a852f393bed5ddffc9132541798e708fac0109a`.
It is **not** claimed to be the entire original working tree.

Unrelated operator/generation scripts and configurations, build metadata,
historical preparation/proxy shell recipes, datasets, scene assets, Blender,
API exchanges, results, credentials and an environment lock are not included.
Published files are unchanged; even historical defaults are retained. Do not
silently edit this versioned directory: a behavior change needs a new identity.

The full source package is kept to preserve lazy imports and packaged resources,
not to designate its generation tools or Nonrect modules as current releases.
This snapshot is selected **only for `single_room`**. It does not repair the
missing rectangular Multi-room historical snapshot, and its publication does
not certify cross-harness compatibility, performance or complete scoring.

This release is repository-local and is not installed into the root project's
wheel. The main package, Nonrect source and running frozen directories are not
overwritten by publication.
