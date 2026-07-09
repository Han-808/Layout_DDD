# OAR Evaluator v0

The Object-Architecture Relationship evaluator is a deterministic geometry proxy for checking object-to-room constraints.

Current architecture input is intentionally small:

```json
{
  "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
  "scene_height": 2.8
}
```

The canonical form is also supported:

```json
{
  "room": {
    "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
    "height": 2.8
  }
}
```

## Supported Relations

OAR v0 supports only floor, wall, and corner proxy checks:

```text
on_floor
against_wall
near_wall
below_wall
at_corner
```

String aliases such as `against south wall`, `near east wall`, `below north wall`, and `at northeast corner` are normalized into canonical relation specs.

## Assumptions

- Floor is the plane `z = 0`.
- Wall is each room boundary edge extruded vertically from `z = 0` to `scene_height`.
- Corner is each room boundary vertex.
- Object geometry is represented by center, size, and rotation.
- Checks use bbox/OBB proxy geometry only.
- There is no mesh contact, wall thickness, wall material, opening, or navigability model.
- `below_wall` is a weak proxy: the object must be near a wall and its top must not exceed `scene_height + eps_z`.

## Unsupported

The following are out of scope for v0 and are reported as skipped:

```text
ceiling
door
window
on_wall
hanging
navigability
region relations
wall-mounted surfaces
floor meshes
wall meshes
openings
materials
```

Unsupported relations do not affect `overall_score`.

## CLI

```bash
python scripts/evaluate_oar.py \
  --scene scene.json \
  --out oar_report.json
```

Optional relation and config overrides:

```bash
python scripts/evaluate_oar.py \
  --scene scene.json \
  --relations oar_relations.json \
  --config oar_config.json \
  --out oar_report.json
```

The report averages all checks that were actually called. Invalid inputs count as called checks with score `0.0`; unsupported relations are skipped.
