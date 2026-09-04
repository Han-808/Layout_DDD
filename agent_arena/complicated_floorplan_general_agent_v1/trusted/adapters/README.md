# Agent adapters

Adapters remain deliberately thin. They select a registered Agent runtime,
record its model/service identity and compute policy at the strongest available
disclosure level, and expose the canonical `TODO.md` without changing it. They
must not add tools, instructions, files, internet access, or database
privileges.

No Agent family is selected here. An entrant may be registered if its adapter
satisfies the same outer Seatbelt, workspace, gateway, prompt, tool, and
artifact contract. Product-specific flags belong in the later entrant
registration, not in the arena's scientific contract.

The canonical task is the byte-identical workspace `TODO.md`; delivery may be
registered as a workspace file, stdin, or an argv path. The adapter may only
transport that task, never rewrite or extend it.

The host credential or account-auth files must never enter the Agent workspace
or environment. If an Agent requires a remote model/service, its adapter must
use a scoped loopback gateway or an equivalently narrow host-owned broker.

An adapter is not authorized for a benchmark run until it passes all of:

1. outer-isolation smoke test;
2. exact executable/version hash registration when locally inspectable;
3. model/service identity and compute-policy disclosure;
4. scoped gateway request/model budget test;
5. one no-evaluator preflight episode;
6. public workspace file-set equality check against the other entrants.
