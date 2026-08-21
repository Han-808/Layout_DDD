# Generation campaign v2

## Current interface

`benchmark.scene_generation.campaign` is the current model-agnostic entrypoint
for API2 and API3 scene generation. It composes reviewed protocol grammars with
public model/campaign profiles, Phase A retrieval profiles, ignored local
bindings, and the unchanged frozen two-stage kernel.

This is an operational source-checkout command. The library wheel includes the
Python API but deliberately does not duplicate the frozen `tools/` core or its
brief/prompt bundle. Invoking the campaign command outside a Layout_DDD source
checkout therefore fails explicitly instead of guessing a repository root.

```bash
python -m benchmark.scene_generation.campaign check --campaign api2-kimi-k3-scene10-v2
python -m benchmark.scene_generation.campaign resolve --campaign api2-kimi-k3-scene10-v2
python -m benchmark.scene_generation.campaign resource-gate --campaign api2-kimi-k3-scene10-v2
python -m benchmark.scene_generation.campaign preflight --campaign api2-kimi-k3-scene10-v2
python -m benchmark.scene_generation.campaign run \
  --campaign api2-kimi-k3-scene10-v2 --output-dir /new/write-once/output
```

All commands accept `--profile-root` and `--retrieval-catalog`. Commands that
need runtime bindings accept `--generation-bindings` and
`--resource-bindings`. Normal selection precedence is explicit argument,
environment-selected binding file, then the ignored repository-local file.

## Composition and supported aliases

The interface dispatches by protocol grammar, never by a model-name substring:

```text
CampaignProfile
  -> ModelProfile
       -> RouteProfile
            -> codec + gateway + option + response grammar
  -> retrieval_profile_id
       -> dataset + encoder + index + retrieval policy
  -> workflow + brief-set + retry + artifact + preflight contracts
```

The checked-in model registry is a reproducibility fixture, not a hardcoded
support whitelist. A new model alias can reuse an existing grammar by adding a
reviewed model profile and campaign; a new wire grammar requires code and
fixtures. Request-option validation fails if a profile supplies a field its
selected codec cannot emit.

## Fail-closed command order

`check` first verifies the complete active bundle inventory (campaign runtime,
all campaign registries, frozen core, retrieval runtime and retrieval catalog),
then parses exact-key JSON, resolves every ID, checks workflow/brief/prompt and
artifact-schema hashes, compiles the complete retry policy, and resolves the
retrieval profile. It reads no local binding or credential and uses no network.

`resource-gate` performs Phase A resource hashes, encoder revision/shape checks,
alignment checks, and the golden Top-1 suite. `preflight` and `run` perform that
same strict gate, reverify every trusted bundle, import the frozen core, verify
the bundles again, and only then read a credential or use the network. `run`
performs a live generic preflight before invoking
`FrozenTwoStageOrchestrator`. Preflight content validation and returned-model
identity are explicit contract/profile fields; neither depends on campaign-ID
prefixes nor model-name suffix heuristics.

The declared retry contract includes the exact retryable transport/HTTP
statuses, delivery-ambiguity policy, retry count/delay, and continue/stop policy.
Semantic and schema retries remain zero inside a one-shot case.

## Local binding boundary

Copy
`configs/generation/campaign_v2/generation_bindings.local.example.json` to
`.runtime/generation_bindings.local.json` and replace only the endpoint and
credential environment name. The credential value remains in that environment
variable or an external secret store. The binding file is ignored by Git.

Generation binding selection is explicit `--generation-bindings`, then
`LAYOUT_DDD_GENERATION_BINDINGS`, then the ignored local default. Retrieval
binding selection uses the equivalent Phase A precedence.

`resolve` proves both logical binding sets exist without reading the credential
value. Its JSON contains logical IDs and a binding identity hash, never an
endpoint, local path, credential environment name, credential, or headers.

## Artifact contract v3

Campaigns opt into `hy34-two-stage-artifacts-v3`. The case payloads and Stage A
-> retrieval -> Stage C ordering remain unchanged. The frozen runner received
additive v3 schema constants and transport-error redaction; the campaign layer
separates the transport-private runtime model from its artifact-safe public
projection:

- private at runtime: endpoint and credential;
- public in `run_manifest.json`: model/profile values, timeouts, retry counts,
  route/source hashes, and `transport_binding=private-redacted-v1`.

The source manifest records registry/content hashes and logical contract IDs.
It contains no endpoint, binding path, credential environment name, headers,
request/response body, prompt text, reasoning, or dynamic request ID. Existing
Existing v1/v2 artifact files are not rewritten; new campaign runs opt into v3.
The controller also verifies the frozen core's declared run/case schema before
execution and re-reads `run_manifest.json` plus every written
`case.result.json` after termination. The
sanitized preflight report is embedded in source provenance and returned by the
CLI; raw provider content is never copied into it.

## Public registry safety and evidence

Public registries reject duplicate/unknown keys, non-finite numbers,
secret/environment/path fields, URL schemes (including gRPC), UNC and Windows
paths, embedded `/Users` and `/home` paths, and host:port literals.

Tests cover legacy request/header characterization, grammar-only alias reuse,
unknown-reference and silent-option rejection, binding precedence/redaction,
resource-gate-before-credential ordering, and loopback preflight, retry,
response, artifact, exit, and redaction behavior. The evaluator is not imported
or modified by this package.
