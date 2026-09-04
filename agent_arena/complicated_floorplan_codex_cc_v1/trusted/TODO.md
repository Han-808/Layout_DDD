# Trusted-side launch TODO

The fixed suite, public Agent workspace, database capability service, outer
Seatbelt executor, and scoped model-gateway primitive are implemented.

Before any paid or official generation:

- [ ] Select the exact Codex and/or Claude Code entrants.
- [ ] Pin each Agent executable version and content hash.
- [ ] Pin each underlying model ID and reasoning policy.
- [ ] Register a Responses-compatible gateway adapter for Codex.
- [ ] Register the verified Claude Code API/gateway adapter, if selected.
- [ ] Set a common model-request budget in addition to the fixed tool budget.
- [ ] Run one isolated, no-evaluator preflight per Agent.
- [ ] Confirm no credential, host-home, repo, other episode, or arbitrary
      network access from inside the Agent process.
- [ ] Freeze the entrant profile hashes.
- [ ] Obtain explicit launch approval.

Do not weaken the network policy or inject host credentials to bypass these
gates.
