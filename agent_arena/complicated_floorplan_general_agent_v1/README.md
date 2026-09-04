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
- `arena.lock.v2.json`: current content hashes for every controlled arena file.
- `arena.lock.json`: preserved predecessor lock from the earlier, overly
  specific draft; v2 hashes it as part of the provenance chain.
- `TODO.md`: canonical Agent-facing task template.
- `fixed_suite/`: approved layouts/programs plus public shared-DB identity.
- `public/`: files copied into each Agent workspace.
- `trusted/`: host-only materializer, verifier, gateway, and isolated executor.
- `trusted/agent_registration.example.json`: participant-neutral registration
  template; the current entrant list is intentionally empty.
- `episodes/`: generated episode folders; contents are intentionally ignored by
  git.

## Safe local checks

```bash
/usr/bin/python3 trusted/verify_arena.py
/usr/bin/python3 -m unittest discover -s tests -v
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
or launched by this package.
