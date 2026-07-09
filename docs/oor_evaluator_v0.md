# OOR Evaluator v0

Lightweight Object-Object Relationship evaluation for generated scene JSON.

## Taxonomy

Supported OOR v0 labels:

- proximity: `near`
- direction_of: `left`, `right`, `in_front`, `behind`, `above`, `below`, `aligned_with`
- attachment: `contact`
- facing: `face_to`
- containment: `within`, `out_of`

Aliases:

- `front` -> `in_front`
- `facing` / `face` -> `face_to`
- `next_to` -> `near`
- `contains(A, B)` inverts subject/object and checks `within(B, A)`

Unsupported in v0:

- `on`
- `under`
- `attached_to`
- `between`
- `around`
- `against`
- room/wall/object-architecture relations

Unsupported labels are skipped and reported; they do not affect the average.

## Input

Objects may be under `objects` or `assets`.

Direct object:

```json
{
  "id": "table_1",
  "jid": "asset-id",
  "category": "table",
  "size": [1.2, 0.8, 0.7],
  "center": [0, 0, 0.35],
  "rotation": [0, 0, 90]
}
```

Pose object:

```json
{
  "id": "table_1",
  "size": [1.2, 0.8, 0.7],
  "pose": {
    "center": [0, 0, 0.35],
    "rotation": [0, 0, 90]
  }
}
```

Canonical relation:

```json
{
  "subject_id": "chair_1",
  "object_id": "table_1",
  "type": "near",
  "category": "proximity",
  "frame": "anchor_object",
  "strength": "hard"
}
```

Relation aliases accepted by the evaluator:

- `subject` for `subject_id`
- `target_id`, `anchor_id`, or `object` for `object_id`
- `relation` for `type`

When relation specs are not provided explicitly, extraction order is:

1. `scene["oor_relations"]`
2. `scene["relations"]`
3. `object["placement_intent"]["relative_relations"]`

For per-object placement intent, the current object id is the subject and
`target_id`/`anchor_id`/`object_id`/`object` is the anchor.

## Deterministic-Only Runtime

OOR v0 never calls a VLM, LLM, remote model, or judge. The API reserves a
`runtime.vlm_fallback` config slot for a future implementation, but v0 only
records whether such fallback was requested. It does not execute it.

Default runtime config:

```json
{
  "runtime": {
    "mode": "deterministic",
    "vlm_fallback": {"enabled": false}
  }
}
```

If a caller sets `runtime.vlm_fallback.enabled = true` or requests a
non-deterministic mode, the report still uses deterministic checks and returns
`runtime.vlm_fallback.status = "not_implemented"`.

## Output

```json
{
  "evaluator_version": "oor_v0",
  "status": "ok",
  "evaluation_mode": "deterministic",
  "runtime": {
    "mode": "deterministic",
    "deterministic_only": true,
    "vlm_fallback": {
      "available": false,
      "requested": false,
      "status": "disabled"
    }
  },
  "overall_score": 0.75,
  "num_checks_called": 4,
  "num_passed": 3,
  "num_failed": 1,
  "checks": [],
  "skipped": [],
  "notes": []
}
```

`overall_score` is the average score of checks with status `checked` or
`invalid_input`. Skipped unsupported relations are excluded.

If no check is called, status is `no_checks_called` and `overall_score` is `0.0`.

## CLI

```bash
python scripts/evaluate_oor.py \
  --scene path/to/scene.json \
  --relations path/to/relations.json \
  --out outputs/oor_report.json
```

`--relations` is optional. `--config` can point to a JSON object that overrides
the default thresholds recursively.

## Geometry Assumptions

- Right-handed coordinates.
- +X is right/east.
- +Y is front/north.
- +Z is up.
- Size is `[width, depth, height]`.
- yaw=0 means the object faces `-Y`.
- yaw positive rotates counter-clockwise around +Z.
- The evaluator uses OBB/bbox proxy geometry from center, size, and rotation.

## Limitations

- No OAR.
- No VLM/LLM calls, fallback, or judge in v0.
- No mesh, Blender, Habitat, physics engine, ray tracing library, or real mesh contact.
- No 3+ object relations such as `between` or `around`.
- `contact` is bbox/OBB surface proximity only.
- Direction checks are anchor-local for object-object relations.
- `above` and `below` require vertical ordering plus XY overlap or closeness.
