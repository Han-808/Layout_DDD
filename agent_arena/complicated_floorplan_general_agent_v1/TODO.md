# Furnish one fixed residential FloorPlan

You are the registered general-purpose Agent assigned to exactly one SIEVE scene.
Complete the task by creating and sealing `submission.json`. Do not merely
describe a solution in prose.

## Current case

- Arena: `{{ARENA_ID}}`
- Scene: `{{SCENE_ID}}`
- Rooms: `{{ROOM_COUNT}}`
- Wall segments: `{{WALL_SEGMENT_COUNT}}`
- Required total instances: `{{TARGET_MIN}}` to `{{TARGET_MAX}}`, inclusive
- Shared asset database: `{{DATABASE_SNAPSHOT_ID}}`

## Authoritative files

- `floorplan.json`: immutable shared-global room polygons and walls.
- `room_program.json`: required room-program multiset and count range.
- `task.json`: frozen case, tool-budget, and evaluation-scope contract.
- `database-interface.json`: exact database tool protocol.
- `submission.schema.json`: final JSON structure.

Do not change the FloorPlan, walls, coordinate frame, room-program multiset, or
count range. Inputs are verified again outside this workspace.

## Database interface

Use only the supplied shared database through:

```text
./sieve-agent-tool get-task
./sieve-agent-tool search-assets --query TEXT --size W D H --top-k K
./sieve-agent-tool inspect-asset ASSET_ID
./sieve-agent-tool validate-submission submission.json
./sieve-agent-tool finalize-submission submission.json
```

Search returns candidates; you choose appropriate assets from the returned
bounded Top-K. Asset identity, category, description, and bbox returned by the
database are authoritative. Do not invent asset IDs or obtain assets from any
other source.

## Composition requirements

- Assign every required room program exactly once and furnish every room.
- Meet the exact inclusive total-instance range.
- In rooms whose area permits, build multiple spatially distinct and
  functionally coherent clusters.
- Include primary anchors, secondary functional objects, and a controlled
  proportion of small supported or wall-attached objects.
- Aim for 25-35% secondary objects without arbitrary clutter or meaningless
  duplication.
- Include at least one meaningful support or attachment relation in every
  eligible room.
- Repeated categories must form an intentional ensemble or have distinct
  roles.
- Preserve realistic scale, circulation, access, support, containment,
  collision avoidance, and functional orientation.
- Keep each object and relation inside its assigned room. Cross-room relations
  are outside the current evaluator scope.

## Final artifact

Write `submission.json` conforming exactly to
`non_rectangular_agent_submission_v1`, containing:

1. the exact `layout_id`;
2. an `object_plan` using the existing
   `non_rectangular_multi_room_object_plan_v2` contract;
3. one ordered `{room_id, slot_id, asset_id}` binding for every planned slot;
4. one shared-global placement using
   `non_rectangular_global_catalog_placement_v1`.

Use `validate-submission` while iterating. When the scene is ready, run:

```text
./sieve-agent-tool finalize-submission submission.json
```

Only a successfully sealed submission is accepted. Evaluator scores, Judge
feedback, and hidden failure labels are unavailable during generation.

The operating system confines you to this episode workspace. Do not attempt to
inspect parent directories, the host home, other episodes, or external
networks.
