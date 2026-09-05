# Pi API compatibility and release gate

This arena supports one API family per experiment and one or more model
profiles from that family. Models and scenes run sequentially. A family-level
kernel lock prevents a second local experiment from using the same API family
at the same time.

The API information in the repository-level `Support/API.md` is the deployment
source of truth. This file describes only the arena-side contract. It contains
no production endpoint or credential.

## Registered families

| Stable family ID | Example model profiles | Host transport |
|---|---|---|
| `api2` | GPT-5.6 Sol, Kimi K3, GLM 5.3, Hy4-preview | Direct scoped gateway; API2 auth/query adapter owned by the host |
| `api3` | Claude Opus 5, Sonnet 5, Fable 5 | Direct scoped gateway; API3 session and strategy headers owned by the host |
| `tokenhub` | Hy4-preview | Pinned host-managed LiteLLM Anthropic adapter, then the scoped gateway |

API4/API5 are deployment-slot labels, not scientific identities. TokenHub
experiments always use the stable `tokenhub` family and route profile IDs.

An experiment cannot mix families or transport adapters. One credential is
acquired once by the host supervisor and reused only for that experiment's
serial model profiles. If two models actually require different credential
contracts, they must be placed in separate experiment files even if their
service is colloquially given the same API number.

## Controlled files

- `profiles/api_families.json` owns credential kind, shared cooldown, route
  membership, and the concurrency ceiling.
- `profiles/route_profiles.json` owns Pi protocol, paths, host auth strategy,
  physical-attempt identity, response contract, and transport adapter.
- `profiles/model_profiles.json` owns wire identity, Pi compatibility,
  reasoning, token/context limits, timeout, retry policy, and accepted response
  model identity.
- `experiments/*.example.json` select one family, one or more registered models,
  the fixed ten scenes, and the common harness limits.
- `profiles/runtime_bindings.example.json` is secret-free. Copy it to the
  gitignored `profiles/runtime_bindings.local.json` and replace only the routes
  used by the selected experiment with operator-approved deployment bases and
  opaque binding revision IDs. Never put a key or Authorization value in it.
- `adapters/tokenhub_litellm_v1.json` pins the exact TokenHub adapter runtime,
  model alias, upstream model class, output limit, thinking block, signed
  reasoning replay bridge, zero inner retries, and timeout layering.

The plan and every episode identity hash the controlled model, route,
transport, Pi runtime, arena, system/task prompt, database, and limit
contracts. Provider endpoints and credentials are deliberately neither
recorded nor hashed; an opaque `binding_profile_id` identifies the approved
deployment revision without disclosing it. The operator MUST increment that ID
whenever the endpoint, deployment, route mapping, entitlement, or other
request-routing behavior changes. Reusing an ID after such a change would make
old sealed outputs appear eligible for resume under a different deployment.

## No-key checks

Run the ordinary tests and arena verifier before any real credential is
entered:

```bash
cd /absolute/path/to/complicated_floorplan_general_agent_v1

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/absolute/path/to/repository/src \
/absolute/path/to/repository/.venv/bin/python -m pytest -q tests

/usr/bin/python3 trusted/verify_arena.py
```

Build and inspect a fully sealed plan without network access, a credential, or
an episode:

```bash
/usr/bin/python3 trusted/run_pi_experiment.py \
  --experiment trusted/experiments/api2-all-models.example.json \
  --runtime-bindings trusted/profiles/runtime_bindings.example.json \
  --dry-run
```

For API2 and API3, the credential-free runtime gate verifies the pinned Pi
runtime and direct transport contract:

```bash
/usr/bin/python3 trusted/run_pi_experiment.py \
  --experiment trusted/experiments/api3-all-models.example.json \
  --runtime-bindings trusted/profiles/runtime_bindings.example.json \
  --verify-runtime-only
```

For TokenHub, the same mandatory gate additionally requires and fully hashes
the pinned LiteLLM source, venv, interpreter bundle, launcher, and lock, then
executes an offline codec check proving that the Anthropic request shape
contains the fixed high-thinking block. It next runs real pinned Pi through
the real pinned LiteLLM proxy and a host-owned raw-provider identity relay into
a loopback fake Anthropic server. The fake server requires the second request
to contain the exact first-turn thinking text, opaque signature, tool-use
ordering, and tool-result identity. The relay verifies the raw Anthropic
`message_start.message.model` before any successful stream is released to
LiteLLM. The same integrated gate also proves fail-closed behavior for wrong or
missing raw identity, wrong successful content type, a truncated identity
preface, and a valid raw provider 2xx that fails later inside LiteLLM response
transformation; then it proves one bounded retry for an explicit provider HTTP
429. A raw-provider-2xx pending/acknowledgement handshake prevents that
normalized local 5xx from becoming a duplicate provider request.
Each ambiguous case must produce exactly one physical gateway attempt, drain
the raw-provider connection, terminate the owned LiteLLM process tree, release
its listener, reverify the exact runtime, and only then start a replacement.
Before its temporary artifacts are removed, the gate scans the actual public
Pi, gateway, relay, and transport records and fails if fixture reasoning,
signatures, credentials, endpoints, raw headers, or raw request/response
artifact names were persisted.
The configured provider route is loopback-only and uses no production
credential or external provider. This is a request-routing contract, not an
OS-level network sandbox for the third-party runtime:

```bash
/usr/bin/python3 trusted/run_pi_experiment.py \
  --experiment trusted/experiments/tokenhub-hy4.example.json \
  --runtime-bindings trusted/profiles/runtime_bindings.example.json \
  --tokenhub-litellm-runtime /absolute/path/to/uni_llm_hhr_cursor_snapshot \
  --verify-runtime-only
```

The local release test can also run the complete gate above. Its upstream calls
go only to benchmark-owned loopback fixtures; it makes no production/external
provider request:

```bash
SIEVE_TOKENHUB_LITELLM_RUNTIME=/absolute/path/to/uni_llm_hhr_cursor_snapshot \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/absolute/path/to/repository/src \
/absolute/path/to/repository/.venv/bin/python -m pytest -q \
  tests/test_managed_transport.py
```

Skipping that optional installed-runtime pytest is acceptable for a developer
machine without the snapshot, but it is not acceptable for an official
TokenHub release gate. The `--verify-runtime-only` command above is the
non-skippable official check: absence or drift of either runtime, startup
failure, inventory mismatch, signed-replay mismatch, or incomplete cleanup
makes it exit nonzero.

## Real route preflight

After the no-key gates pass, use the hidden terminal prompt. The credential is
read once for the whole experiment; it is not accepted as a command-line
value:

```bash
/usr/bin/python3 trusted/run_pi_experiment.py \
  --experiment trusted/experiments/api2-all-models.example.json \
  --runtime-bindings trusted/profiles/runtime_bindings.local.json \
  --preflight-only
```

For TokenHub, add the pinned adapter runtime argument. `--preflight-only`
starts the private loopback adapter and runs a real, streaming, two-turn tool
call through pinned Pi. It validates model identity, response envelope, tool
arguments, tool result continuation, reasoning-state replay when required, and
stream termination. It creates no scene submission.

For unattended execution, `--credential-env NAME` reads and removes `NAME`
from the supervisor process environment. The parent shell has its own
environment copy, so the operator must still `unset NAME` there after the
command. Never place the credential value in argv, a JSON/YAML file, a log, or
an Agent workspace.

## Official execution

An official run additionally requires the local shared-database resource
binding file:

```bash
/usr/bin/python3 trusted/run_pi_experiment.py \
  --experiment trusted/experiments/api3-all-models.example.json \
  --runtime-bindings trusted/profiles/runtime_bindings.local.json \
  --resource-bindings /absolute/path/to/retrieval_bindings.local.json \
  --execute
```

Execution first inspects the exact sealed model-by-scene matrix. If every exact
episode already exists, it acquires no credential, starts no transport, and
runs no live preflight. For a partial resume, only models with missing scenes
receive a live preflight; exact sealed scenes are reused and never overwritten.
A present but invalid or mismatched episode is terminal and is not silently
replaced.

The API2 GPT/Kimi/GLM deployment contract currently does not provide a trusted
response-model echo, so their public routing assurance is explicitly
`request_profile_only_unverified_response`: the harness proves the selected
route and wire model but does not claim a positive server-returned identity.
TokenHub Hy4 and the registered API3 profiles require the configured response
identity contract. If an API2 deployment later exposes a trustworthy echo,
that must be introduced as a new controlled profile rather than silently
changing the current assurance level.

## Retry and ambiguity contract

The scoped gateway is the only request-retry owner. Pi/provider retries and the
managed LiteLLM retry loop are disabled.

- A connection failure proven to occur before request bytes may leave the host
  is retryable.
- Explicit HTTP `408`, `409`, `425`, `429`, `500`, `502`, `503`, and `504` are
  retried after the shared 30-second family cooldown.
- Each logical model request has one initial attempt and at most five such
  infrastructure retries.
- Any failure after request transmission, while establishing response
  identity, or during a response stream is ambiguous and terminal for that
  logical request. It is never blindly re-sent. The gateway checks a private
  managed-transport ambiguity sideband before deciding whether a normalized
  local 5xx is retryable.
- After a TokenHub post-send/stream ambiguity, the raw-provider relay is
  poisoned. No later logical unit may reuse it. The supervisor must abort and
  drain its raw-provider/downstream sockets, terminate the birth-bound LiteLLM
  tree, prove both loopback listeners released, reverify the sealed runtime,
  and start a new LiteLLM process before continuing.
- The entire Agent episode has one attempt. A process timeout or nonzero exit
  does not replay prior model calls.
- A final exhausted retryable response still triggers the shared cooldown so
  the next scene/model does not immediately hammer the same family.

TokenHub's pinned LiteLLM timeout is 1860 seconds, at least 30 seconds longer
than the outer 1800-second gateway timeout. This guarantees that an unknown
post-send wait reaches the outer ambiguous boundary before LiteLLM can convert
its own timeout into a local HTTP error. The adapter is health-checked through
its authenticated exact `/v1/models` inventory before every preflight/model
episode boundary.

## TokenHub reasoning contract

The host gateway injects and audits `reasoning_effort=high` and validates
reasoning-state replay across tool turns. Because the pinned LiteLLM version
does not recognize the custom Hy4 alias as a built-in reasoning model, the
managed adapter separately fixes the exact Anthropic provider field:

```json
{"type":"enabled","budget_tokens":4096}
```

The offline codec gate proves that this field and `max_tokens=65536` reach the
Anthropic request transform and that the OpenAI `reasoning_effort` field itself
is not forwarded as an unsupported provider parameter. LiteLLM 1.83.14 emits
the Anthropic replay signature under `thinking_blocks`, while Pi 0.85 persists
only `reasoning_details`. The scoped gateway therefore performs a strict,
in-memory conversion from signed `thinking_blocks` to a benchmark-owned
`reasoning_details` format on the response, and restores the exact signed block
before the next LiteLLM request. Unsigned, malformed, foreign-format, or
conflicting replay data fails closed.

The credential-free fake-upstream gate proves byte-preserving transport and
ordering, but cannot prove that a fixture signature is cryptographically valid
for TokenHub. The paid `--preflight-only` gate is the second boundary: TokenHub
must return a real signature and accept it on the following tool-result turn.
Signature deltas are concatenated in stream order and exposed to Pi only as one
complete signed normal-thinking block. Pi 0.85 cannot safely retain independent
signatures for multiple normal-thinking blocks, so that shape fails closed;
ordered redacted-thinking blocks remain discrete and are replayed unchanged.
Only booleans, counts, hashes, and stable failure codes are persisted. Hidden
reasoning and signature bytes may be held in memory only as needed for the next
tool turn; they are not written to public reports, gateway events, transcripts,
or scene artifacts.

## Isolation scope

Pi receives only a short-lived loopback gateway capability, never a provider
credential or provider endpoint. Agent Bash processes do not inherit the model
gateway capability. They do inherit only the narrow per-episode database Unix
socket path and capability required by `sieve-agent-tool`; the broker still
enforces the frozen snapshot, query budget, and method contract. The pinned
extension registers each Bash PID and blocks before sending the first command
byte until the host acknowledges the exact birth-bound process identity. The
host applies Seatbelt, resource ceilings, workspace quotas, process-tree
cleanup, sanitized logs, hash-chained tool transcripts, and host-side
submission sealing.

macOS Seatbelt is inherited by descendants and protects files and network even
if a process changes session. macOS does not provide this runner with a cgroup-
equivalent atomic descendant set, however. A deliberately hostile participant
could attempt a very fast double-fork plus `setsid` between host snapshots. Do
not claim VM-grade hostile-process containment from this local mode. Run
untrusted/adversarial Agent implementations in a disposable VM or equivalent
per-episode host boundary while keeping the same prompts, database broker,
gateway, and artifact contract.
