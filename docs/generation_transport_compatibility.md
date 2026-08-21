# Generation Transport Compatibility Layer

Status: proposed architecture; documentation only. Implementation requires a
separate approval.

## Decision summary

The compatibility refactor is limited to scene generation. It formalizes the
boundary between the frozen two-stage generation workflow and model-provider
wire protocols.

More specifically, this proposal applies to the standalone frozen runners under
`tools/` used by the Kimi-K3, GLM-5.3, and Opus 4.8 campaigns. It does not merge
or replace the separate public `layout-ddd-generate` path, its adapter registry,
or `OpenAICompatibleModel`. Unifying those paths would change transport, retry,
artifact, and security semantics and therefore requires a separate proposal.

It does **not** change the evaluation pipeline. Generated scenes continue to be
materialized into the same canonical artifacts and are then consumed by the
existing evaluator without importing or calling the new transport layer.

The unit of reuse is a provider route, not a model. A provider route composes a
wire codec with a gateway/authentication policy:

- a model using an already-supported route adds a configuration entry;
- a genuinely new request/response envelope adds one codec;
- a genuinely new gateway or authentication contract adds one gateway policy;
- a different generation algorithm or stage graph requires a separately
  versioned workflow and explicit review.

Consequently, Kimi-K3, GLM-5.3, Claude Opus 4.8, and future models should not
each require a new copy of the generation runner.

## Current state

The current active runners already share most workflow behavior through
`tools/api3_anthropic_runner_v2/generation_runner.py`. That core owns the
following sequence:

1. load and validate a brief;
2. make the Stage A object-plan request;
3. validate the first Stage A emission;
4. perform the frozen deterministic retrieval;
5. construct the canonical Stage C generation input;
6. make the Stage C placement request;
7. validate the first Stage C emission;
8. write the case result, audit manifest, run manifest, and summary.

The model-specific runners currently adapt the core by importing it from an
exact filesystem path and replacing module-level functions:

- `tools/api2_kimi_k3_runner_v1/kimi_k3_generation_runner.py` replaces request
  construction and headers for API2 Chat Completions;
- `tools/api2_glm53_runner_v1/glm53_generation_runner.py` also replaces response
  extraction for API2 Responses;
- `tools/api3_opus48_thinking_runner_v1/generation_runner.py` replaces request
  construction to add the frozen adaptive-thinking settings on the API3
  OpenAI-compatible Chat Completions route.

This proves that a shared workflow boundary already exists. The structural
problem is that the boundary is implicit: it uses `sys.path`, dynamic imports,
module-global mutation, and model-specific runner copies. The proposed layer
makes that boundary explicit and testable.

Kimi and Opus illustrate why "same protocol" needs a precise definition. Both
use a Chat Completions-shaped body, but API2 and API3 use different auth/header,
session, option, and preflight policies. They can share the Chat Completions
codec without pretending that their complete provider routes are identical.

The initial route matrix is:

| Model campaign | Wire codec | Gateway policy | Model-specific code after migration |
| --- | --- | --- | --- |
| Kimi-K3 | OpenAI Chat Completions | API2 | Config and fixtures only |
| GLM-5.3 | OpenAI Responses | API2 | Config and fixtures only |
| Claude Opus 4.8 High | OpenAI Chat Completions | API3 | Config and fixtures only |

## System boundary

```text
briefs + model config
          |
          v
  GenerationRunSpec
          |
          v
 frozen two-stage orchestrator
   | Stage A          ^ normalized response
   | retrieval        |
   | Stage C          |
   v                  |
 protocol provider ---+
          |
          v
 frozen generation artifacts
 (`generation_input.json` + `catalog_placement_v1.json` + provenance)
          |
          v
 existing catalog adapter/materialization
          |
          v
 canonical generated scene / trusted dataset
          |
          v
 existing evaluation pipeline (unchanged)
```

The compatibility layer ends when canonical generation artifacts have been
written. It is not part of evaluation.

The following evaluation components are explicitly out of scope:

- `scripts/run_camera_cal_scene_level.py`;
- `src/benchmark/api/evaluation.py`;
- `src/benchmark/evaluator/`;
- `src/benchmark/visual_judge/`;
- camera selection and camera rendering;
- Judge, Discovery, Grouping, and placement prompt text or versions;
- scoring weights, metric ownership, aggregation, and publishability rules;
- the order and semantics of evaluation calls.

No evaluation module should import a generation provider. Evaluation continues
to consume files through the existing canonical dataset/artifact boundary.
`src/benchmark/api/submission.py` already enforces the same direction by
preparing trusted native output for evaluation without running or importing a
generator.

The expected implementation touch set is therefore limited to new modules under
`src/benchmark/scene_generation/frozen_two_stage/`, generation configs,
generation-only tests, and—only after parity is proven—the existing standalone
generation runner shims. The public generation adapter registry is also out of
scope. A change to an evaluator, visual-judge, camera, scoring, or
evaluation-prompt file is a scope violation for this proposal.

## Proposed package structure

The new implementation should live under the existing scene-generation domain
rather than introduce a second top-level workflow system:

```text
src/benchmark/scene_generation/
  interfaces.py                 # existing public interface; unchanged here
  frozen_two_stage/
    spec.py                     # immutable GenerationRunSpec and model config
    orchestrator.py             # frozen Stage A -> retrieval -> Stage C graph
    retry_policy.py             # explicit retry classification and limits
    artifact_layout.py          # canonical paths, manifests, and provenance
    providers/
      base.py                   # codec, gateway, and composed-route contracts
      codecs/
        openai_chat.py          # Chat Completions encode/decode
        openai_responses.py     # Responses encode/decode
      gateways/
        api2.py                 # API2 auth/header/session policy
        api3.py                 # API3 auth/header/session policy
      routes.py                 # typed codec + gateway + option/preflight profile
    compatibility/
      legacy_runner.py          # helpers for existing CLI/Bash entrypoints
```

The existing `SceneGenerator` interface and public generation adapters remain
unchanged in this phase. The new internal package may later expose a compatible
facade, but it must not be registered into `layout-ddd-generate` as part of this
proposal.

## Component responsibilities

### GenerationRunSpec

`GenerationRunSpec` is immutable, serializable, and safe to record in a public
manifest after credential fields have been excluded. It identifies:

- the ordered brief set;
- the model configuration key and wire model;
- the provider key;
- generation parameters such as output-token limit and reasoning effort;
- retry policy;
- output root and artifact schema versions;
- hashes of prompts, schemas, briefs, retriever configuration, and runner
  sources.

It does not contain prompt text, provider code, evaluation settings, or a
literal credential.

### Provider route

A provider route is the typed composition of:

- a codec that owns the request and response envelope;
- a gateway policy that owns authentication, endpoint, header, and session
  behavior;
- an allowlisted option/preflight profile for fields such as thinking or
  reasoning effort.

Together they own only provider-specific behavior:

- construct the request envelope from the exact system prompt and canonical
  user value supplied by the orchestrator;
- construct safe transport metadata and inject credentials from the named
  environment variable at runtime;
- send one request through the existing transport primitive;
- normalize a provider response into content, optional reasoning metadata,
  usage metadata, HTTP status, and a safe transport classification;
- perform protocol/model-identity preflight checks that do not change the
  generation stage graph.

A route must not own briefs, prompts, retrieval, placement validation, artifact
eligibility, scoring, or evaluation logic.

Codec, gateway, and route methods must be instance-scoped. They must not mutate
module globals, patch another module, or depend on import order.

### Two-stage orchestrator

The orchestrator owns the existing generation semantics and must preserve them:

```text
Stage A first emission
  -> strict object-plan validation
  -> frozen deterministic Top-1 retrieval
  -> canonical generation_input.json
  -> Stage C first emission
  -> strict placement validation
  -> terminal case result
```

The orchestrator receives a provider route instance. It has no model-name branches;
all differences in wire envelopes are delegated to the provider, and all
differences in values such as model ID or reasoning effort come from config.

### Retry policy

Retry behavior is data, not copied runner control flow. The policy records:

- maximum infrastructure retries;
- retryable transport classifications;
- retryable HTTP statuses;
- retry delay;
- whether ambiguous timeouts may be retried;
- whether a terminal failed case permits the next brief to continue.

Semantic or schema retries remain disabled unless a separately versioned
workflow explicitly authorizes them. Fresh-case recovery campaigns remain an
outer workflow and do not silently alter the one-shot contract.

### Artifact layout

Artifact layout preserves the current filenames, schema versions, write-once
behavior, eligibility rules, and provenance semantics. At minimum, compatibility
is required for:

- `run_manifest.json`;
- `execution_policy.json`;
- `fixed_instruction.json`;
- `scene_request.json`;
- Stage A and Stage C attempt directories;
- validation and retrieval artifacts;
- `catalog_placement_v1.json`;
- `audit_manifest.json`;
- `case.result.json`;
- `summary.json`.

Existing output roots remain valid evaluator inputs.

### Legacy compatibility shim

Existing Python entrypoints and Bash launchers remain available for at least
one compatibility cycle. A shim may translate current command-line arguments
and model config into `GenerationRunSpec`, then invoke the shared orchestrator.

The shim must not reimplement request construction, retry loops, stage logic, or
artifact writing. Existing command names, arguments, exit codes, output paths,
and safe terminal summaries remain stable during migration.

## Model onboarding rules

Use the following decision table when adding a model:

| Change | Required work | New model-specific runner? |
| --- | --- | --- |
| New model ID on an existing codec + gateway + option/preflight profile | Add a model config entry and fixtures | No |
| Different token limit, effort, or supported optional field already declared by that route | Change config | No |
| Different preflight expectation but identical codec/gateway behavior | Add an allowlisted declarative preflight profile | No |
| New request/response envelope | Add one codec and fixtures | No copied workflow runner |
| New gateway or authentication contract | Add one gateway policy and fixtures | No copied workflow runner |
| Different Stage A/retrieval/Stage C semantics | Add a separately versioned workflow after explicit review | Possibly a new workflow, not a transport runner |

Model aliases, endpoints, credentials, limits, and supported optional fields
belong in configuration. Provider selection is explicit; it must not be inferred
from a model-name substring.

"Configuration only" is valid only when codec, gateway/auth behavior, request
option grammar, response grammar, retry policy, preflight policy, and artifact
policy are already supported. Configuration must not become an unrestricted
request-template language that hides new protocol code.

## Frozen compatibility invariants

The migration is acceptable only if the old and new paths are equivalent for
each active route. The following are frozen:

1. Stage A and Stage C prompt bytes and SHA-256 values.
2. Brief bytes, schemas, coordinate conventions, and retrieval configuration.
3. Stage order: Stage A, validation, retrieval, Stage C, validation.
4. First-emission and one-shot semantics.
5. Request JSON bytes for an equivalent model configuration, excluding only
   explicitly nondeterministic transport fields such as session IDs.
6. Required safe headers and provider-specific request fields.
7. Response normalization and schema-error classification.
8. Retryable conditions, maximum attempts, delay behavior, timeout ambiguity
   policy, and continue-after-case behavior.
9. Eligibility, terminal status, and stop-batch decisions.
10. Artifact names, JSON schemas, field semantics, summaries, and exit codes.
11. Credential redaction and the rule that credentials are read from the
    environment rather than stored in specs or artifacts.
12. The canonical output consumed by dataset preparation and evaluation.

Source-code hashes, runner version identifiers, timestamps, and their derived
audit hashes are expected to change when implementation source changes. They
must be explicitly versioned and remain traceable; they must not be forged to
match the old runner. Content hashes for unchanged prompts, briefs, schemas,
retriever inputs, request bodies, first emissions, and frozen placements remain
the parity evidence.

No parity exception may be introduced merely to simplify the provider API. Any
intentional behavioral change requires a new versioned workflow and separate
approval.

## Verification gates

Before a legacy shim delegates to the new implementation, automated parity
tests must cover the old and new paths with frozen local fixtures.

Required tests:

- request-envelope byte parity for API2 Chat Completions, API2 Responses, and
  API3 OpenAI-compatible Chat Completions;
- response normalization for success, malformed JSON, incomplete response,
  empty content, and provider error envelopes;
- transport and HTTP retry classification, including ambiguous timeouts;
- Stage A -> retrieval -> Stage C call-order parity;
- validation and terminal-status parity;
- `case.result.json`, `summary.json`, and manifest semantic parity after removal
  of documented nondeterministic and intentionally versioned provenance fields;
- source, prompt, schema, and config hash checks;
- legacy CLI argument, exit-code, and output-root compatibility;
- a no-import test proving evaluator modules do not depend on generation
  provider modules;
- credential redaction and public-manifest separation checks.

Tests must use local fixtures and fake transports. They must not consume a live
API merely to establish structural parity.

## Known compatibility traps

- The current adapters patch module-global hooks. Loading two routes in one
  process can overwrite behavior, so the new API must be instance-scoped before
  any concurrent registry is introduced.
- The shared core currently recognizes more retryable HTTP statuses than the
  Kimi/GLM execution-policy files and preflight summaries declare. Migration
  must first characterize and preserve the actual runtime decisions. Aligning
  the declaration and behavior is a separate, explicitly reviewed change.
- Ambiguous transport delivery is not retryable in the formal one-shot path;
  recovery runners opt into it. The compatibility layer must not collapse those
  policies.
- The current audit layout preserves raw request/response and optional reasoning
  artifacts while redacting credentials from persisted headers. Migration must
  preserve that artifact policy unless a separate security change is approved.
- `src/benchmark/models/OpenAICompatibleModel` is not a drop-in replacement for
  the frozen generation transport: its retry, serialization, and artifact
  semantics differ. The provider layer must initially retain the frozen
  transport behavior.

## Migration sequence

1. Characterize and freeze the current request, response, retry, and artifact
   behavior for Kimi-K3, GLM-5.3, and Opus 4.8.
2. Add immutable specs and the provider protocol without changing legacy
   entrypoints.
3. Move the shared Stage A -> retrieval -> Stage C logic behind the typed
   orchestrator while retaining byte-parity tests against the frozen core.
4. Implement the three protocol providers and pass fixture parity.
5. Convert each existing runner into a thin compatibility shim, one route at a
   time.
6. Run existing runner checks, local fixture tests, and end-to-end artifact
   parity gates.
7. Keep old entrypoints and source bundles for at least one compatibility
   cycle. Deletion or deprecation requires a separate decision.

At every step, the old path remains available until the corresponding new path
passes all gates.

## Non-goals

This refactor does not:

- change prompts or improve generated scene quality;
- change retrieval ranking, query construction, or asset selection;
- introduce semantic retries or select the best result by score;
- merge or rescore experimental outputs;
- change evaluation, rendering, camera evidence, VLM judging, grouping, or
  leaderboard logic;
- remove historical runners or artifacts;
- force unrelated scene-generation systems onto this two-stage workflow.

## Expected outcome

After migration, adding a model that uses one of the supported provider routes
is a configuration change plus fixture coverage. The repository retains one
versioned two-stage generation workflow, reusable wire codecs and gateway
policies, stable legacy launchers, canonical artifacts, and the same independent
evaluation pipeline.
