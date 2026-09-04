# Agent adapters

Adapters remain deliberately thin. They select an exact Agent executable and
version, exact model, and host-created loopback gateway, then pass the canonical
`TODO.md` through stdin. They must not add tools, instructions, files, internet
access, or database privileges.

For Codex, the registered command must use non-interactive `codex exec`, an
ephemeral session, explicit workspace-write sandboxing, no approvals,
`--ignore-user-config`, `--ignore-rules`, and `--skip-git-repo-check`. A custom
Responses-compatible provider should point only to the scoped loopback gateway.
The host API key or `~/.codex/auth.json` must never enter the Agent workspace or
environment.

Claude Code must follow the same outer Seatbelt, workspace, gateway, and tool
contract. Its exact CLI flags must be registered only after that executable is
installed and its version-specific non-interactive behavior has been verified.

An adapter is not authorized for a benchmark run until it passes all of:

1. outer-isolation smoke test;
2. exact executable/version hash registration;
3. exact model and reasoning-policy registration;
4. scoped gateway request/model budget test;
5. one no-evaluator preflight episode;
6. public workspace file-set equality check against the other entrants.
