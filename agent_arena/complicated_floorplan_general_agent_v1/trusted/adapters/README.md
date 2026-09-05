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

The common Pi host adapter, API-family profiles, credential-free runtime gate,
and real streaming tool-call preflight are specified in
`../API_COMPATIBILITY.md`. A profile or example experiment is registration, not
authorization to contact a provider; a live route requires an explicit
operator action and a gitignored runtime binding.

For TokenHub, authorization also requires the credential-free real-Pi →
real-LiteLLM → host identity relay → fake-Anthropic signed-thinking replay
gate. The host relay, not LiteLLM's rewritten OpenAI stream, proves the raw
Anthropic response model before releasing success. The gate additionally
proves that unsafe post-send/stream failures cannot be converted into a retry
by LiteLLM: the private ambiguity sideband makes the request terminal, poisons
the relay, drains the raw connection, recycles the owned proxy, reverifies the
pinned runtime, and only then permits the next logical unit. A first-turn codec
check or post-LiteLLM model alias alone is insufficient for a multi-turn
tool-using Agent.
