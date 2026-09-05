# SceneEval-100 HY4 online capture runner

This directory is the MNET generation/capture side of the agreed reproduction
protocol. It preserves every HTTP attempt and does not parse, repair, or
evaluate the generated layout JSON.

## Frozen scope

- Model: Hy4-T3-A49B-DSA-1M-SFT0730-Opus5.
- Dataset: ordered SceneEval IDs 0..99 selected under the user-approved
  human-authored subset rule.
- Sanitized source rows contain only id and description. ID is runner-side
  provenance and is never sent to the model.
- prompt_protocol.txt is the complete system message.
- The user message is exactly the original SceneEval Description.
- The system prompt asks the model to infer room count, room types, room
  dimensions, and room placement, using a global non-negative coordinate
  system.
- The response layout contract contains rooms and objects; each object
  references its containing room.
- No layout JSON parsing, schema validation, code-fence removal, extraction,
  repair, critique, or reasoning-channel normalization occurs during MNET
  generation.

## Frozen client configuration

openai_clients.yaml is the authoritative client/request configuration:

- base URL: the approved internal OpenAI-compatible endpoint
- configured model: openai/Hy4-T3-A49B-DSA-1M-SFT0730-Opus5
- wire model: Hy4-T3-A49B-DSA-1M-SFT0730-Opus5
- timeout: 1800 seconds
- max_retries: 2 for consecutive infrastructure/API failures
- temperature: 0.9
- top_p: 1.0
- top_k: -1
- max_tokens: 65536
- repetition_penalty: 1.0
- reasoning_effort: high
- preserved_thinking: true

LiteLLM is not installed in the approved runtime. The runner therefore maps
the YAML to direct OpenAI-compatible HTTP while preserving each physical
request as a separate attempt. It does not use hidden library retries.

Every physical HTTP request has a fresh UUID SessionID plus
StrategyType: ConsistentHash. Exact request headers are stored in
request-headers.json.

## Retry and minimum-output policy

For a complete usable 2xx API envelope, the runner computes:

~~~text
visible_output_tokens =
  usage.completion_tokens
  - usage.completion_tokens_details.reasoning_tokens
~~~

- visible_output_tokens >= 10: captured; finish the scene.
- visible_output_tokens < 10: short_output; preserve the entire attempt and
  retry without a count limit.
- Missing, non-integer, negative, or inconsistent token usage:
  token_count_unavailable; preserve the response and stop the batch. The
  runner never estimates token counts.
- transport_ambiguous: never retry; stop the batch.
- transport_failure, http_error, or invalid_api_response: retry at most twice
  consecutively, matching max_retries: 2. A third consecutive failure becomes
  retry_exhausted and the batch continues to the next scene.
- A successful short_output resets the consecutive infrastructure-failure
  counter.

Every retry waits 30 seconds. Every attempt receives a new SessionID and lives
in its own attempt_NN directory. An interrupted attempt is conservatively
finalized as transport_ambiguous and is never resent.

## Input

The input JSONL must contain exactly 100 ordered rows with only these fields:

~~~json
{"id":0,"description":"The original SceneEval description"}
~~~

Validate the input without contacting HY4:

~~~bash
/root/miniconda3/envs/lf/bin/python /hcccao/sceneeval_repro/runner/run.py check-input --input-jsonl /hcccao/sceneeval_repro/data/sceneeval_human100_generation.jsonl
~~~

## Artifacts

Run root:

- run-manifest.json
- runner-source-manifest.json with runner version and SHA256 for every Python
  source file used by the runner
- input.snapshot.jsonl
- prompt_protocol.txt
- layout.schema.json
- openai_clients.yaml
- execution-summaries/summary_*.json: one immutable summary for every normal
  run or resume invocation

Each scene_NNN directory:

- scene.result.json
  - for captured scenes this directly records the accepted attempt number,
    request hash, response hash, raw-content hash, SessionID, and X-Request-Id
- retry_attempt_NN.scheduled.json for each scheduled retry
- attempt_01 and any later attempt_NN directories

Each attempt directory:

- request.json
- request-headers.json
- attempt.started.json
- api-response.body and response-headers.json when received
- raw-content.txt for exact message.content
- logs/reasoning.txt for exact message.reasoning when supplied
- logs/reasoning_content.txt for exact message.reasoning_content when supplied
- capture.json with byte hashes, token counts, and explicit no-normalization
  flags
- attempt.result.json

reasoning and reasoning_content are preserved independently and never compared
or merged. The runner never creates normalized-content.txt or validation.json.

## Tests

The tests use a loopback fake server only. They do not contact HY4:

~~~bash
cd /hcccao/sceneeval_repro/runner
/root/miniconda3/envs/lf/bin/python -m unittest discover -s tests -v
~~~

## Local ordered multi-model suite

`run_local_multimodel.py` reuses this runner's exact 100-row input, system
prompt, per-scene user prompt, retry lifecycle, and immutable capture
artifacts. Its only compatibility differences are at the provider boundary:

- fixed serial model order: `claude-opus-4-8`, `kimi-k3`, `gpt-5.6-luna`;
- model-specific thinking fields: Claude Opus 4.8 uses adaptive thinking with
  `output_config.effort: high`, while Kimi and Luna use top-level
  `reasoning_effort: high` (no HY4 sampling/template fields);
- no MNET routing headers; and
- a runtime-only bearer token which is redacted from all artifacts.

Validate locally without a request:

~~~bash
python tools/sceneeval_hy4_runner_v050/run_local_multimodel.py check-input
~~~

Smoke-test the exact first SceneEval case once on each model before the full
200-scene suite. This is a real paid API probe and writes a separate immutable
artifact tree:

~~~bash
python tools/sceneeval_hy4_runner_v050/run_local_multimodel.py smoke-probe \
  --scene-id 0 \
  --output-dir output/sceneeval_local_smoke_probe_v050
~~~

Start a new run. The API key prompt is hidden and the key is never stored:

~~~bash
python tools/sceneeval_hy4_runner_v050/run_local_multimodel.py run \
  --output-dir output/sceneeval_local_multimodel_v050
~~~

Resume the same immutable run without resending any prior attempt:

~~~bash
python tools/sceneeval_hy4_runner_v050/run_local_multimodel.py run \
  --output-dir output/sceneeval_local_multimodel_v050 \
  --resume
~~~
