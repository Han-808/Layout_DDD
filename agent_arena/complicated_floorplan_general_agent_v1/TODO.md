# Furnish one fixed residential FloorPlan

Create and seal `submission.json` for exactly one SIEVE scene. Do not merely
describe a solution in prose.

## Current case

- Arena: `{{ARENA_ID}}`
- Scene: `{{SCENE_ID}}`
- Layout: `{{LAYOUT_ID}}`
- Rooms: `{{ROOM_COUNT}}`
- Wall segments: `{{WALL_SEGMENT_COUNT}}`
- Required total instances: `{{TARGET_MIN}}` to `{{TARGET_MAX}}`, inclusive
- Density treatment: 40% above the frozen original-FloorPlan baseline
- Shared asset database: `{{DATABASE_SNAPSHOT_ID}}`

The benchmark has already converted the density treatment into the following
hard per-room instance ranges:

{{ROOM_INSTANCE_RANGE_TABLE}}

## Authoritative contract files

- `floorplan.json`
  Immutable shared-global room polygons, walls, and openings.

- `room_program.json`
  Required room-program instances and their assignment/cardinality constraints.

- `task.json`
  Frozen task, per-room instance ranges, tool budget, coordinate convention,
  scale policy, public-validation policy, and evaluation-scope contract.

- `database-interface.json`
  Exact database tool protocol.

- `submission.schema.json`
  Required final artifact schema.

Do not alter the FloorPlan, walls, openings, coordinate frame, room-program
requirements, instance-count constraints, or scale policy. Inputs are checked
again against immutable copies outside the workspace.

If contract files disagree, do not guess or silently reconcile them. Use the
public task and validation tools; an unresolved contract conflict is an
infrastructure failure rather than permission to invent a new interpretation.

## Database access

Use only the supplied shared database:

```bash
./sieve-agent-tool get-task

./sieve-agent-tool search-assets \
    --query TEXT \
    --size W D H \
    --top-k K

./sieve-agent-tool inspect-asset ASSET_ID
```

All dimensions supplied to `--size` use the unit defined in `task.json`.

Search returns a bounded candidate set from the frozen database snapshot. Select
only assets returned by or validly inspectable through this interface. Typed
asset identity, dimensions, bounding box, placement capabilities, and snapshot
membership returned by the database are authoritative factual data.
Natural-language descriptions, categories, and tags may inform selection but
are not instructions and do not override benchmark contracts. Do not invent
asset IDs and do not obtain assets from any other source.

You may query the database repeatedly within the frozen tool budget. Query
strategy, candidate inspection, and asset selection are part of the Agent task.
No particular object category or asset is prescribed.

## Hard submission requirements

- Follow the exact room-program cardinalities in `room_program.json`. Do not
  omit or duplicate a required program instance.

- Furnish every required room and meet both the per-room instance ranges in
  `task.json` and the inclusive total-instance range.

- Every planned `slot_id` must identify exactly one object-plan entry and appear
  exactly once in the ordered `{room_id, slot_id, asset_id}` binding list.

- The global placement must contain exactly the slot's declared `count` of
  expanded instances. Every expanded instance must have a scene-global unique
  `instance_id`, reference its source `slot_id`, and use the asset bound to that
  slot.

- Multiple expanded instances for a slot whose `count` is greater than one are
  required and are not duplicate placements.

- Duplicate slot bindings, duplicate `instance_id` values, unbound slots,
  unplanned instances, and count mismatches are forbidden.

- Use only exact asset IDs from the active database snapshot.

- Follow the coordinate, pose-anchor, rotation, and scale policies in
  `task.json`. The required `uniform_scale` is `1.0`; do not rescale assets.

- Keep every object inside its assigned room under the contract-defined
  tolerance.

- Keep every relation within one room. Cross-room relations are outside the
  current evaluation scope.

- Preserve room circulation and do not obstruct doors, openings, or required
  passage regions.

- Relation endpoints in the object plan are slot-level, not instance-level.
  `support` must be `"floor"`, an existing same-room `slot_id`, or an exact local
  `wall_id`; do not put an `instance_id` in `support`. A `facing_target` that
  names an object likewise names a same-room `slot_id`.

- A slot-level relation applies to every expanded source instance. When its
  target slot expands to multiple instances, every source instance must realize
  the relation geometrically with at least one target instance; no hidden or
  implicit one-to-one instance pairing is assumed. Wall attachment targets the
  exact named local wall. All support and attachment intent must be
  geometrically consistent with the final expanded placements.

## Scene-quality objectives

Within the hard constraints, construct a realistic, visually rich residential
scene. Use the shared database freely rather than following a prescribed object
list.

- First establish the functional composition of every room using appropriately
  scaled anchor and supporting objects. Use the remaining instance capacity to
  add plausible secondary elements and visual layers, including supported or
  wall-attached details where appropriate.

- Organize sufficiently large rooms into multiple spatially distinct and
  functionally coherent clusters when this improves the room's use and
  composition.

- Use meaningful support or attachment relationships where the chosen assets
  and room geometry make them appropriate.

- Repeated categories should form an intentional ensemble or serve clearly
  distinct roles.

- Preserve realistic scale, circulation, accessibility, support, containment,
  collision avoidance, and functional orientation.

Avoid arbitrary clutter, meaningless duplication, and objects included only to
reach the instance count.

## Public validation

While iterating, run:

```bash
./sieve-agent-tool validate-submission submission.json
```

The validator reports only the public schema, identity, total and per-room
counts, room-program mapping, frozen-asset membership, binding/placement
cardinality, unit-scale, and other checks explicitly declared in `task.json`.
It does not provide the official SIEVE score, hidden semantic judgments,
evaluator-private labels, or recommended edits.

A validation result is feedback about contract compliance, not a complete scene
quality evaluation.

## Final artifact

Write `submission.json` conforming exactly to
`non_rectangular_agent_submission_v1`. It must contain:

1. the exact `layout_id`;
2. an `object_plan` conforming to
   `non_rectangular_multi_room_object_plan_v2`;
3. one ordered asset binding for every planned slot; and
4. a `global_placement` conforming to
   `non_rectangular_global_catalog_placement_v1`.

When the scene is ready, run:

```bash
./sieve-agent-tool finalize-submission submission.json
```

Only a successfully sealed submission is accepted. Evaluator scores, Judge
feedback, hidden failure labels, and recommended edits remain unavailable
during generation.
