# Metrics Spec v0

The benchmark compares explicit bbox-based 3D room layouts with a small set of scalar metrics. Model outputs remain JSON layouts; meshes, CAD, Blender code, and Habitat scenes are not benchmark outputs.

## Per-Case Metrics

Each case writes `case_metrics.json` with exactly these comparable scalar fields, plus identifiers:

- `validity_gate`: bool
- `room_consistency_score`: int or null, range 0..4
- `room_consistency_score_norm`: float or null, range 0..1
- `object_presence_rate`: float or null, range 0..1
- `specified_relation_pass_rate`: float or null, range 0..1
- `specified_attachment_pass_rate`: float or null, range 0..1
- `primary_score`: float, range 0..1

There is no `constraint_score` in v0.

## Validity Gate

`validity_gate` is false only for catastrophic failures:

- output JSON cannot parse
- layout schema or bbox geometry is invalid
- bbox size is non-positive, NaN, or infinity
- all placed objects are completely outside the room boundary
- most or all objects fully overlap into essentially one volume
- for structured cases, all required objects are missing

Minor collision, one partially out-of-bound object, small floating gaps, imperfect attachments, and questionable spacing are debug evidence only. They do not fail the gate.

## Room Consistency

`room_consistency_score` is produced by the room-level judge using room-level rendered views:

- `topdown_room`
- `front_room`
- `corner_room`

Rubric:

- 0: unusable or not a plausible room
- 1: severe semantic or layout problems
- 2: partially plausible room but many issues
- 3: mostly coherent room, minor issues
- 4: coherent, natural, functional, and matches the task well

`room_consistency_score_norm = room_consistency_score / 4.0`.

## Object Presence

`object_presence_rate` is null for `prompt_only`.

For `structured_basic` and `structured_relation`:

```text
placed_required_objects / required_objects
```

v2 cases prefer id-based matching. If only categories exist, matching falls back to category counts.

## Explicit Relations And Attachments

`specified_relation_pass_rate` is null if no explicit visible relations exist. Otherwise:

```text
passed_visible_explicit_relations / total_visible_explicit_relations
```

`specified_attachment_pass_rate` is null if no explicit visible attachments exist. Otherwise:

```text
passed_visible_explicit_attachments / total_visible_explicit_attachments
```

Only items with `visible_to_model=true` are evaluated. Pair judges output pass or fail only: no numeric score, partial score, or uncertain class.

## Primary Score

If `validity_gate` is false, `primary_score = 0.0`.

Otherwise, `primary_score` is the equal-weight mean of non-null active metrics:

- `prompt_only`: `room_consistency_score_norm`
- `structured_basic`: `room_consistency_score_norm`, `object_presence_rate`
- `structured_relation`: `room_consistency_score_norm`, `object_presence_rate`, `specified_relation_pass_rate` if non-null, `specified_attachment_pass_rate` if non-null

Example:

```text
mean(0.75, 1.0, 0.5, 1.0) = 0.8125
```

## Benchmark Outputs

Benchmark runs write `benchmark_metrics.csv` with columns:

```text
case_id,model,input_level,validity_gate,room_consistency_score,room_consistency_score_norm,object_presence_rate,specified_relation_pass_rate,specified_attachment_pass_rate,primary_score
```

They also write `benchmark_summary.json`, grouped by `input_level`, with:

- `num_cases`
- `primary_score_mean`
- `primary_score_std`
- `validity_gate_rate`
- `room_consistency_score_norm_mean`
- optional metric means when applicable

Debug physical and spatial flags are not benchmark CSV columns in v0.
