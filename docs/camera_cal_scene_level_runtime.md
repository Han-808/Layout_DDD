# Camera-cal scene-level runtime structure

Status: E1 mechanical extraction. The authoritative compatibility command
remains `scripts/run_camera_cal_scene_level.py`.

The package `benchmark.camera_cal_scene_level` owns only leaf mechanics:

- `io.py`: JSON/YAML reads, atomic JSON writes, hashes, and UTC timestamps;
- `progress.py`: progress JSONL persistence and terminal formatting;
- `telemetry.py`: API-call accounting, token normalization, metric telemetry,
  and the mechanical API-call tracker.

The historical runner keeps the same public class/function names as
compatibility facades. Existing imports and monkeypatch points therefore
continue to work. The runner still owns case discovery, scheduling, model and
renderer wiring, `run_case`, report construction, and every semantic policy.

E1 does not change Judge prompts, camera selection, rendering, metric weights,
deductions, evaluation order, retry policy, report schemas, or output paths.
The frozen E0 contract records the pre-extraction runner blob and semantic
source hashes; focused tests compare the facade and leaf implementations.
Evaluation Campaign source identity explicitly hashes every tracked Python
module under this package. Adding or changing a runtime leaf therefore changes
the campaign protocol fingerprint and makes an older resume fail closed.

Later structural phases must remain separately approved. In particular, E1
does not authorize moving `run_case` or the promptless/camera/scene-quality
policy functions.
