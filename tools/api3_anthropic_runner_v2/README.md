# API3 Anthropic paired-10 runner

This runner is a transport-only adaptation of the frozen HY4 SFT0730 Opus5-AW
two-stage runner. The paired briefs, Stage A and Stage C prompts, schemas,
retriever runtime, semantic validation, one-shot policy, and artifact layout are
unchanged. The only experimental changes are the API3 endpoint, credential
source, and four model aliases.

The Anthropic route uses the API3 OpenAI-compatible endpoint
`http://21.214.33.175:4000/v1/chat/completions`. ForgeAX documentation may be
used as a protocol-shape reference only; its endpoint and credential are not
used by this runner.

The API credential is read from `API3_API_KEY` at runtime. It must never be
written to the runner, output artifacts, logs, or command history.

`launch_all_local_api3_failed_retry_max2.sh` is a separate post-main wrapper.
It derives each model's failed public brief IDs only after a terminal 10/10
main summary, launches fresh isolated retry1 outputs, and launches retry2 only
for cases still failed or unprocessed after retry1. It never resumes, overwrites,
or creates a third semantic retry.

`launch_local_api3_retry_watchdog.sh` starts a credential-inheriting background
watchdog immediately. It waits without competing until every main run has a
terminal 10/10 summary, then invokes the max-2 retry launcher. A model with no
failed cases exits without creating retry outputs.
