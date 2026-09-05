# Generation Retrieval Profiles v2

## Purpose

The generation retriever is one shared implementation under
`benchmark.scene_generation.retrieval`. Dataset identity, encoder identity,
index identity, retrieval policy, and local storage are no longer collapsed
into a model-specific runner directory.

This boundary is generation-only. It does not change evaluation, cameras,
Judge prompts, metric definitions, scoring, Stage A/Stage C prompts, query
construction, cosine scoring, size soft scoring, stable tie order, or Top-1
selection.

## Four reviewed descriptors

`configs/retrieval/profiles_v2.json` contains four independently reusable
declarations:

1. A **dataset descriptor** names the asset namespace, metadata collection and
   order fields, and maps dataset-native fields to the canonical asset
   contract. The v2 loader intentionally supports one topology only: a JSON
   object-valued asset map plus a separate ordered-ID array. Other source
   topologies require an explicit reviewed adapter; field mapping alone does
   not make arbitrary metadata shapes executable.
2. An **encoder descriptor** names an implementation contract, arbitrary
   upstream model ID, immutable revision, expected dimension, deterministic
   query settings, package versions, and a complete content hash manifest.
3. An **index descriptor** binds a dataset and encoder to content-addressed
   metadata and matrix resources, expected rows/dimension/dtype, and stable
   order semantics.
4. A **retrieval profile** composes exactly one dataset, encoder, and index with
   the numerical policy and a content-addressed golden suite.

There is no dataset-name, model-name, or provider alias whitelist. Another
dataset×encoder combination adds descriptors and evidence. A new executable
encoder/index algorithm requires reviewed code because configuration is not an
unrestricted plugin language.

## Local logical resources

Tracked profiles contain logical resource IDs, sizes, hashes, revisions, and
dimensions. They contain no absolute path, endpoint, credential, or API key.

A local binding file has schema `generation_resource_bindings_v2` and maps only
a logical resource ID to a local path. It is a registry and may be a superset
covering many dataset×encoder profiles. A selected profile requires only its
own three bound resources; provenance reports only the IDs and hashes actually
consumed by that profile. Selection order is deterministic:

1. explicit `--resource-bindings`;
2. the file named by `LAYOUT_DDD_RETRIEVAL_BINDINGS`;
3. ignored `.runtime/retrieval_bindings.local.json`.

The runtime never scans `$HOME`, a Hugging Face cache, or a mount point. The
checked-in `resource_bindings.example.json` is safe to copy. `.runtime/` is a
shared repo-local state directory; only its `.gitignore` is tracked.

Local paths are never written to public retrieval provenance. Manifests record
profile IDs, catalog/profile hashes, resource content hashes, encoder revision,
dataset/index identities, and the shared runtime source hash.

## Fail-closed validation

Before a generation credential is read or a provider request is made, the
runtime verifies:

- catalog/profile schema and cross-references;
- index dimension equals encoder dimension;
- matrix rows equal dataset asset count and ordered-ID count;
- matrix dtype, finiteness, byte size, and SHA-256;
- metadata byte size and SHA-256;
- unique ordered IDs, exact asset-map alignment, and record identity;
- canonical asset fields and finite positive 3D sizes;
- every encoder snapshot file by relative name, byte size, and SHA-256;
- package contract and golden Top-1/score behavior.

The configured-run trust gate also hashes the complete shared retrieval source
bundle. After runtime initialization and before any credential is read, the
loaded catalog hash, profile hash/ID, consumed resource hashes, and runtime
source identity must exactly match the static trusted view. This closes the
trust-check/load interval instead of trusting a dynamic import by path.

Hugging Face snapshot files may be symlinks into the local blob store. The
runtime follows them and trusts the resulting bytes, not the symlink target
path. Official profiles are always strict: package or golden drift is terminal.

## Current profile and compatibility

The current parity profile is:

```text
imaginarium-qwen3-embedding-0.6b-stable-top1-v2
```

It composes the existing 2,043-row Imaginarium index with the exact Qwen3
Embedding 0.6B revision and six-query golden suite. All six observed Top-1 IDs
and scores match the frozen runtime exactly.

Current generation configs use `frozen_two_stage_run_config_v2`, explicitly
name the shared runtime root, and refer to this profile. The v1 run-config
reader remains available. The tracked v1 `retriever_runtime.py`,
`run_retriever.sh`, `retriever_runtime.pod.json`, and
`retriever_golden_v1.json` remain byte-for-byte frozen replay interfaces.
Current generation calls the shared v2 runtime directly and does not consume
the v1 Pod paths.

The official strict gate is reproducible only with the exact package contract
declared by the encoder profile. The `retrieval` project extra pins NumPy,
Sentence Transformers, and Transformers to those exact versions; it is not a
loose convenience dependency.

## Commands

Static configuration validation requires neither resources nor credentials:

```bash
PYTHONPATH=src:. .venv/bin/python -m \
  benchmark.scene_generation.frozen_two_stage check \
  --run-config configs/generation/api2_kimi_k3_scene10_v2.json
```

Strict resource/golden validation:

```bash
PYTHONPATH=src:. .venv/bin/python -m \
  benchmark.scene_generation.retrieval \
  --catalog configs/retrieval/profiles_v2.json \
  --profile imaginarium-qwen3-embedding-0.6b-stable-top1-v2 \
  --resource-bindings .runtime/retrieval_bindings.local.json \
  gate
```
