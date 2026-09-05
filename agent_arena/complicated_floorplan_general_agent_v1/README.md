# SIEVE Complicated FloorPlan Agent Arena

This folder is the self-contained experiment package for the general-purpose
tool-using Agent track. It does not preselect an Agent implementation or model.
It is not the pipeline-compatible scene-generation-system track.

## Trust boundary

Never start an Agent in this folder or in `fixed_suite/` or `trusted/`.
`trusted/create_episode.py` creates a fresh, single-case folder under
`episodes/<agent>/<scene>/<run>/workspace`; that exact `workspace` directory is
the Agent's only filesystem scope.

The trusted host owns:

- the frozen ten-scene FloorPlan suite;
- the frozen Imaginarium shared-database identity and real resource paths;
- the database service, validation, sealing, normalization, and artifacts;
- model credentials and any scoped model gateway;
- the OS-level isolation policy.

The Agent sees only:

- `TODO.md`;
- one `floorplan.json` and one `room_program.json`;
- `task.json`, `database-interface.json`, and `submission.schema.json`;
- the standalone `./sieve-agent-tool` client;
- files it creates in its own workspace.

It cannot legitimately see the repository, host home, other cases, previous
outputs, evaluator, hidden labels, raw asset resources, or credentials. The
macOS executor is fail-closed and denies network by default. A real model run
must use one explicitly scoped loopback model gateway; broad internet access is
not an accepted fallback.

## Folder map

- `arena.json`: frozen scientific and isolation contract.
- `arena.lock.v5.json`: current content hashes for every controlled arena file.
- `arena.lock.v4.json`, `arena.lock.v3.json`, `arena.lock.v2.json`, and
  `arena.lock.json`: preserved predecessor locks; the
  current lock extends rather than rewrites the provenance chain.
- `TODO.md`: canonical Agent-facing task template.
- `fixed_suite/`: approved layouts/programs plus public shared-DB identity.
- `public/`: files copied into each Agent workspace.
- `trusted/`: host-only materializer, verifier, gateway, and isolated executor.
- `trusted/agent_registration.example.json`: participant-neutral registration
  template; the current entrant list is intentionally empty.
- `trusted/pi_harness/`: frozen common Pi prompt, tool/resource policy, and Pi
  registration template. Installing the harness does not select a model.
- `trusted/pi_harness.py`: creates the per-episode, secret-free Pi model config
  for the one scoped gateway and returns the exact command, TODO stdin, prompt
  hashes, runtime hashes, model settings, budgets, database identity, validator
  policy, and starting-workspace hash.
- `trusted/API_COMPATIBILITY.md`: no-key release gates and the exact one-API,
  many-model credential, transport, retry, reasoning, resume, and isolation
  contract for API2, API3, and TokenHub.
- `trusted/api_profiles.py`, `trusted/profiles/`, and `trusted/experiments/`:
  controlled family/route/model profiles, secret-free runtime-binding
  templates, and example serial experiment matrices.
- `trusted/managed_transport.py` and `trusted/tokenhub_identity_relay.py`:
  verify and own the pinned private TokenHub LiteLLM lifecycle, keep the real
  provider credential out of LiteLLM, prove raw provider model identity, and
  drain/recycle ambiguous transport state; direct API2/API3 routes bypass them.
- `trusted/run_pi_experiment.py`: dry-run, credential-free runtime gate, real
  Pi tool-call preflight, safe resume, and official experiment supervisor.
- `episodes/`: generated episode folders; contents are intentionally ignored by
  git.

## Safe local checks

```bash
/usr/bin/python3 trusted/verify_arena.py
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/absolute/path/to/repository/src \
/absolute/path/to/repository/.venv/bin/python -m pytest -q tests
```

Create one episode without launching an Agent:

```bash
/usr/bin/python3 trusted/create_episode.py \
  --agent-id agent-example \
  --scene-id scene_012121 \
  --run-id dryrun-001
```

Run the isolation smoke test (no model/API request):

```bash
/usr/bin/python3 trusted/smoke_isolation.py
/usr/bin/python3 trusted/smoke_gateway_isolation.py
```

Query the real frozen database through the isolated public interface (still no
model/API request):

```bash
/Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python \
  trusted/smoke_real_database.py \
  --resource-bindings /Users/han_mohan/Desktop/Layout_DDD/.runtime/retrieval_bindings.local.json
```

Every selected Agent adapter must pin the exact runtime version/hash, model or
service identity when available, reasoning/compute policy, command, wall-clock
budget, and scoped gateway configuration. No entrant or generation is selected
or launched merely by installing this package. The included Pi profiles and
experiment JSON files are controlled examples; a real route is used only after
an operator supplies a gitignored runtime binding and explicitly chooses
`--preflight-only` or `--execute`.

API2, API3, and TokenHub setup and release checks are documented in
`trusted/API_COMPATIBILITY.md`. In particular, one experiment is restricted to
one API family but may run several registered models sequentially using the one
credential acquired by its host supervisor.

The local macOS boundary is strong for registered cooperative Agents but is not
a VM-grade containment claim against an intentionally hostile rapid daemonizer.
Use a disposable VM or equivalent per-episode host for adversarial entrants;
the scientific prompt, database, scoped gateway, and sealed-artifact contracts
remain unchanged.

## Fixed complexity treatment

The approved v2 arena keeps the ten FloorPlans and 42 rooms unchanged and sets
object density to 1.40 times the original multi-room FloorPlan benchmark's
planned density envelope. The resulting frozen suite range is 914-1133
instances. Each episode's `task.json` and rendered `TODO.md` contain hard
per-room ranges derived by area-proportional largest-remainder apportionment;
their lower and upper bounds reproduce the scene-level bounds exactly.

No specific asset or object category is prescribed. The prompt asks the Agent
to establish functional room compositions first and use remaining capacity for
plausible secondary elements and visual layers. Semantic role proportions are
not represented as a prompt-fidelity score.

The public validator enforces total/per-room counts and exact unit scale in
addition to the existing schema, identity, asset, binding, placement, and
room-program contracts. The database service writes a hash-chained transcript
of complete public tool calls and results to trusted host-side storage, outside
the Agent's visible and writable workspace.

Successful finalization also writes a trusted host-side seal containing the
validated submission hash. Before normalization, the collector revalidates the
submission and requires its bytes and workspace finalization record to match
that seal. A post-finalization workspace edit therefore fails closed.

## Installed common Pi harness

The local portable runtime is installed outside the sealed arena at
`../runtime_bundles/pi-0.85.0`. Its local manifest records the Node/Pi identity
and bundle fingerprint. Both Chat Completions and Responses routes are
supported, but a concrete wire model, route, reasoning policy, timeout, request
budget, and credential must still be registered and pass a tool-call preflight
before any generation is authorized.
