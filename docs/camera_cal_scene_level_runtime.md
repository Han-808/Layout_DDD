# Camera-cal scene-level runtime structure

Status: E5 final structure. The historical compatibility command remains
`scripts/run_camera_cal_scene_level.py`; the package entrypoint is
`python -m benchmark.camera_cal_scene_level`.

The complete package-owned execution chain is:

```text
benchmark.camera_cal_scene_level.__main__
  -> composition.orchestrator_dependencies()
  -> orchestrator.run_main(deps=...)
  -> case_runtime.run_case_impl(deps=...)
  -> adapters.build_adapters(factories=...)
  -> benchmark.api.evaluation.run_evaluate(...)
```

Run-level discovery, planning, endpoint preflight, scheduling, reports, and
artifact writes remain in the shared package orchestrator. Each case passes
through the explicit `CaseRuntimeDeps` groups (`io`, `resume`, `policy`, and
`external`); adapter construction receives fresh `AdapterFactories`. No
module-level runtime dependency snapshot is used.

The package modules own the following boundaries:

- `cli.py`: frozen argument surface and validation helpers;
- `discovery.py`: ready-case discovery and input-path resolution;
- `planning.py`: route, renderer, control, model, and experiment-plan logic;
- `policy.py`: promptless request/profile, scene-quality, and asset policy;
- `resume.py`: strict completed-case resume eligibility;
- `scheduling.py`: parallel scheduling and route-abort signaling;
- `comparison.py`: pure human/model scene-level comparison;
- `reports.py`: terminal records, resolution audits, and summaries;
- `persisted_scoring.py`: read-only case/run score projections from persisted
  reports for selection, merge, comparison, and viewer consumers; it never
  invokes the evaluator or fills missing coverage;
- `leaderboard_scoring.py`: the separate versioned post-hoc projection used by
  the public scene-generation leaderboard. It applies the provisional web
  weights and deduction multipliers to frozen metric ledgers without changing
  evaluator reports, Judge behavior, cameras, or publishability;
- `provenance.py`: route projection and case-input fingerprints, including the
  concrete extracted runtime source hashes that prevent cross-version resume;
- `observability.py`: API/evidence/render observation wrappers;
- `adapters.py`: injected external model, judge, selector, renderer, and
  evidence-provider construction;
- `audit.py`: optional post-hoc audit-graph projection;
- `case_runtime.py`: single-case orchestration and the one
  `run_evaluate` call;
- `composition.py`: package defaults for concrete evaluator/runtime wiring;
- `orchestrator.py`: package run-level composition and execution;
- `__main__.py`: package CLI entrypoint.

The script remains a dynamic compatibility façade. Its public symbols,
signatures, CLI behavior, and monkeypatch points remain available. On every
call it reads current runner globals to construct orchestration dependencies,
`CaseRuntimeDeps`, and `AdapterFactories`; the package never imports
`scripts.run_camera_cal_scene_level`.

The adapter construction order is frozen:

```text
model configs
-> grouping build/observe
-> judge build/observe
-> selector build/observe
-> renderer
-> L1 provider
-> L3 provider
-> functional probe
-> deterministic selector
-> final renderer
-> preview renderer
```

Judge prompts, camera acquisition, metric/scoring policy, retry behavior,
`run_evaluate` kwargs and call count, report schemas, output paths, and write
ordering are semantic contracts. E0 parity fixtures and fixed-clock trace
tests remain the gate for any change to those contracts.

Source-checkout behavior is the full E5 runtime gate: both the historical
script and package entrypoint must execute the same orchestration and produce
the same canonical artifacts under the same injected/fixed runtime. An
installed wheel has a narrower packaging gate in E5: package import and
`python -m benchmark.camera_cal_scene_level --help` must work without a
checkout or `scripts/` tree. A real installed-wheel evaluation additionally
requires externally supplied dataset, Blender, credentials, and compatible
source/provenance resources; that environment-dependent execution is not
claimed by the help/import gate and must fail closed when those resources are
absent.

Evaluation Campaign source identity hashes the tracked runtime package and
its dependency closure. Any runtime-leaf, composition, policy, or provenance
change therefore changes the protocol fingerprint and makes an older resume
fail closed. HTML viewer rendering is not evaluation or selection logic and
is intentionally outside that identity; the package-owned persisted-scoring
projection used by selection remains inside it.

The canonical evaluator score and the scene-generation leaderboard score are
different named projections. `benchmark_score_100` remains the frozen
evaluator result. The web leaderboard uses
`scene_generation_leaderboard_web_v1` (36% physical plausibility, 50%
functional semantics, and 14% visual coherence) and records its own profile
hash; changing that post-hoc profile does not require re-running evaluation.
