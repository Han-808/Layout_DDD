# Camera-cal scene-level runtime structure

Status: E4 mechanical extraction. The authoritative compatibility command
remains `scripts/run_camera_cal_scene_level.py`.

The package `benchmark.camera_cal_scene_level` owns only leaf mechanics:

- `io.py`: JSON/YAML reads, atomic JSON writes, hashes, and UTC timestamps;
- `progress.py`: progress JSONL persistence and terminal formatting;
- `telemetry.py`: API-call accounting, token normalization, metric telemetry,
  and the mechanical API-call tracker.
- `cli.py`: the frozen argument surface and validation helpers;
- `discovery.py`: ready-case discovery and evaluator input path resolution;
- `planning.py`: route, renderer, control, and experiment-plan construction;
- `resume.py`: strict completed-case resume eligibility;
- `scheduling.py`: parallel scheduling and the shared route-abort signal.
- `comparison.py`: pure human/model scene-level comparison projections;
- `reports.py`: terminal case records, resolution audits, and run summaries;
- `provenance.py`: redacted route projection and case-input fingerprints.
- `adapters.py`: external model, judge, selector, renderer, and evidence
  provider construction only; all concrete builders and the observed-wrapper
  factories are injected by the compatibility façade;
- `case_runtime.py`: the single-case runtime implementation and its explicit
  dependency groups. It owns orchestration around the existing evaluator but
  does not redefine prompts, camera policy, metric policy, scoring, or
  `run_evaluate` semantics.

The historical runner keeps the same public class/function names as dynamic
compatibility facades. Existing imports and monkeypatch points therefore
continue to work: each `run_case` call reads the current runner globals and
constructs fresh runtime dependencies and adapter factories. No package
module may import `scripts.run_camera_cal_scene_level`; the dependency
direction is package runtime → injected concrete implementations, with the
script façade remaining the compatibility boundary.

E1–E4 do not change Judge prompts, camera selection, rendering, metric weights,
deductions, evaluation order, retry policy, report schemas, output paths, or
the `run_evaluate` kwargs/call contract. The adapter construction order is
also frozen: model configs, grouping observe, judge build/observe, selector
build/observe, renderer, L1 provider, L3 provider, functional probe,
deterministic selector, final renderer, and preview renderer.
The frozen E0 contract records the pre-extraction runner blob and semantic
source hashes; focused tests compare the facade and leaf implementations.
Evaluation Campaign source identity explicitly hashes every tracked Python
module under this package. Adding or changing a runtime leaf therefore changes
the campaign protocol fingerprint and makes an older resume fail closed.

Promptless evaluation, camera acquisition, metric/scoring policy, and the
single `run_evaluate` call remain semantic contracts rather than new package
implementations. Any later change to those sources or to dependency wiring
requires the corresponding parity trace and provenance/hash gates.
