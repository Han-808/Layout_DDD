# Repository component and version lifecycle

Layout_DDD contains current runtime code, compatibility shims, frozen
scientific definitions, historical replay code, and local experiment tools.
A numeric suffix alone does not determine whether a component is obsolete.
The authoritative registry for versioned runtime/replay components and
versioned tracked artifacts is
`configs/repository/component_lifecycle_v1.json`.

## Lifecycle states

| State | Meaning | May active code depend on it? | Deletion policy |
| --- | --- | --- | --- |
| `active` | Current supported runtime/config | Yes | No |
| `compatibility` | Supported legacy entrypoint backed by an active replacement | Yes | Only after all callers migrate |
| `frozen_replay` | Immutable historical behavior or scientific definition | No active dependency; explicit replay tests are allowed | Archive review required |
| `local_only` | Present only in a developer worktree or separate local repository | No | First migrate, separate, or quarantine |
| `quarantine` | Disconnected for an observation period with tracked evidence | No | Not yet |
| `deletion_candidate` | Quarantine evidence exists and all checked inbound references are zero | No | Separate explicit approval required |
| `deleted` | Immutable tombstone for an approved deletion; code/artifact bytes are absent | No | Tombstone itself may not be removed or edited |

`frozen_replay` is not a synonym for obsolete. For example, canonical metric
profile v1 and v2 encode different scientific definitions. The v1 profile is
therefore replay material even when no current runtime selects it.

## Required evidence

Run the lifecycle checker from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/check_component_lifecycle.py
```

The checker is static and fail-closed. It does not import component code or
inspect experiment outputs. It verifies:

- exact, non-overlapping component roots;
- owner, reason, replacement, entrypoint, and source declarations;
- consistency between physical files and the Git index;
- exact Git-index blobs materialized by object ID, without checkout/smudge
  filters, so unstaged or locally transformed worktree content cannot hide a
  bad staged revision;
- regular-file Git modes for runnable/replay roots (undeclared symlinks and
  gitlinks are rejected);
- complete coverage of the active generation bundle manifest;
- declared, imported, and literal-path dependencies for both `active` and
  runnable `compatibility` components;
- absence of tracked tests that import or dynamically load untracked
  `tools.*` implementations;
- no runnable dependency on `frozen_replay`, `local_only`, `quarantine`, or
  `deletion_candidate` components;
- classification of every tracked path whose name carries a `vN` version;
- frozen component identities over path, Git mode, and bytes; frozen artifact
  byte hashes plus explicit Git mode; packaged-resource mirror byte equality;
  and
- tracked quarantine evidence plus zero checked inbound references before a
  component can be marked `deletion_candidate`;
- an immutable `deleted` tombstone with the candidate's content hash and
  tracked explicit-approval evidence after physical deletion; and
- baseline-to-current transition checks that reject disappeared IDs, changed
  roots/paths, edits to frozen records, skipped quarantine/candidate stages,
  destructive metadata drift, and tombstone edits.

The lifecycle registry deliberately records major untracked component roots.
Those roots may be absent in a clean clone. Their declaration prevents tracked
runtime/tests from silently depending on files that happen to exist on one
machine.

## Clean-checkout gate

The strongest repository-level check runs against a temporary local clone of
one committed Git revision:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/check_clean_checkout.py --source-ref HEAD
```

The gate materializes only the selected commit's `git archive`; it creates a
fresh index with the source repository's object format, then verifies every
path, mode, and blob object ID against the selected commit. Archive transforms
such as `export-subst` therefore fail rather than silently changing tested
bytes. The snapshot has no remote origin and does not copy unreachable Git
objects or untracked files. Subprocesses receive a filtered environment with
no inherited credential variables and `PIP_NO_INDEX=1`. This is not an
operating-system network sandbox, so the report says
`network_isolation_enforced=false` rather than claiming stronger isolation.
The gate also supplies the selected commit's first-parent lifecycle registry
as the transition baseline. It then runs the component lifecycle checker,
active runner source inventory, complete tracked pytest collection, the
deterministic canonical suite (external-data/Blender/loopback tests are
excluded), and an installed-wheel public API/resource smoke test.

When the caller's worktree differs from the selected revision, a successful
result is labelled `ok_ref_only`. This means the commit is reproducible; it is
not a claim about uncommitted changes. Use `--require-worktree-match` only when
the entire worktree is expected to be clean.

## Deletion workflow

Deletion is a separate reviewed change. The required sequence is:

1. Migrate active callers to a declared replacement.
2. Remove Python, shell, config, test, entrypoint, and active-manifest inbound
   references.
3. Move the component to `quarantine`, record its ISO start date and minimum
   observation days, add tracked evidence identifying the component/root, and
   pin the exact path/mode/content identity. That identity and metadata remain
   immutable through the destructive transition chain.
4. Re-run the clean-checkout gate.
5. Change the registry state to `deletion_candidate`; the checker must prove
   zero checked inbound references and pin the candidate content hash.
6. Present the exact file list for explicit deletion approval.
7. Delete the bytes, change the entry to `deleted`, switch a component to
   `no_tracked_files`, and retain its ID, original root/path, pinned hash,
   quarantine history, and tracked approval evidence. Never remove or edit the
   tombstone; the parent-registry transition gate enforces this.

Historical run artifacts are not scanned by this checker because they can be
large and may contain protected model data. A deletion review must separately
confirm that required replay source hashes and provenance have been preserved.
