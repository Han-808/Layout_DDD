# Support Material

`Support/` contains everything that is useful to the project but is not part of
the core Python workflow, its schemas/configs, or its Python supporting tools.

```text
Support/
├── Assets/                 local asset library
├── Scenes/                 source and converted scenes
├── datasets/               calibration datasets
├── data/                   external and derived data
├── docs/
│   ├── reference/          architecture and metric notes
│   └── research/           plans, handoff, experiments, and progress
├── notes/                  literature and historical working notes
├── bash/
│   ├── mnet/               MNET launchers
│   └── hyak/               Hyak launchers
├── artifacts/
│   ├── outputs/            experiment outputs
│   ├── reports/            generated reports
│   └── result/             historical result material
├── legacy/                 retired benchmark compatibility code
└── third_party/            external source snapshots and archives
```

The local repository root does not retain aliases or symlinks for these
directories. Local code and tests resolve them explicitly through `Support/`.
The MNET checkout keeps its existing remote `Assets/` and `outputs/` storage
contract; that remote mount layout is independent from this local organization.

New Bash launch commands use `Support/bash/...`. Python entry points remain in
`scripts/`, and the core package remains in `src/benchmark/`.

Large/support-only directories are excluded from Git. Do not package
`Support/Assets`, `Support/Scenes`, `Support/artifacts`, `Support/data`,
`Support/datasets`, `Support/notes`, or `Support/third_party` into code bundles.
