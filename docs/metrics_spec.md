# Metrics Spec v1

The workflow compares explicit bbox-based 3D room layouts with VLM-as-judge scores over rendered evidence. Model outputs remain JSON layouts; meshes, CAD, Blender code, and Habitat scenes are not benchmark outputs.

## Per-Case Metrics

Each case writes `case_metrics.json` with exactly these comparable scalar fields, plus identifiers:

- `validity_gate`: bool
- `room_consistency_score`: int or null, range 0..4
- `room_consistency_score_norm`: float or null, range 0..1
- `object_presence_rate`: float or null, range 0..1
- `specified_relation_pass_rate`: float or null, range 0..1
- `specified_attachment_pass_rate`: float or null, range 0..1
- `primary_score`: float, range 0..1

There is no `constraint_score` in v1.

## Validity Gate

`validity_gate` is false only when model output cannot be converted into any parseable layout scene:

- output JSON cannot parse into an object
- generation fails before any layout scene exists

Schema issues, malformed objects, invalid bbox geometry, non-positive sizes, room-boundary issues, below-floor objects, above-wall-height objects, serious collisions, imperfect attachments, and questionable spacing are debug evidence only. They do not fail the gate directly for parseable layouts.

## Room Consistency

`room_consistency_score` is produced by the VLM judge using rendered view evidence:

- `topdown_global_xy`
- each object group's `xy`, `yz`, and `xz` views

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

v2 cases prefer id-based matching. If only categories exist, matching falls back to category counts. This metric is reported for debugging and comparison but is not averaged into `primary_score` in v1.

## Explicit Relations And Attachments

`specified_relation_pass_rate` is null if no explicit visible relations exist. Otherwise:

```text
passed_visible_explicit_relations / total_visible_explicit_relations
```

`specified_attachment_pass_rate` is null if no explicit visible attachments exist. Otherwise:

```text
passed_visible_explicit_attachments / total_visible_explicit_attachments
```

Only items with `visible_to_model=true` are evaluated. The VLM judge outputs pass or fail for each explicit relation/attachment.

## Primary Score

If `validity_gate` is false because no parseable scene exists, `primary_score = 0.0`.

Otherwise, `primary_score` is the VLM room/layout score:

```text
primary_score = room_consistency_score_norm
```

Relation, attachment, and object-presence rates remain reported scalar diagnostics.

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

Debug sanity, renderability, physical, and view flags are not benchmark CSV columns in v1. They are stored in `evaluation_report.json` under `debug_evidence`.
