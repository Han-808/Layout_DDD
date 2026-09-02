Non-rectangular resilient evaluation runner
===========================================

The runner is additive and only evaluates through
``evaluation_mode="non_rectangular_multi_room"``.  It never calls or changes
the rectangular materializer/evaluator route.

Launcher
--------

Use one ``--model-root MODEL=PATH`` option for each available generation
cohort, in the intended model order.  Partial cohorts are accepted; only scene
directories containing all required generation artifacts enter strict
preflight.

For the established API2 GPT-5.6-Sol Judge path, use ``--api2-gpt56``.  The
runner prompts for ``APP_ID:APP_KEY`` when ``STANDARD_API_CREDENTIAL`` is not
already set, generates an ephemeral local master key, owns the Azure-standard
localhost proxy, restarts it before later room attempts when needed, and
removes the credential from runner state on exit.

.. code-block:: bash

   PYTHONPATH=src .venv/bin/python \
     scripts/run_non_rectangular_resilient_evaluation.py \
     --api2-gpt56 --port 4010 \
     --model-root claude-opus-5-aihub=/abs/generation/claude-opus-5-aihub \
     --output-root Support/artifacts/outputs/non_rectangular_evaluation/opus5-api2 \
     --asset-csv /abs/Support/Assets/imaginarium_asset_info.csv \
     --asset-root /abs/Support/Assets/imaginarium_assets \
     --catalog-snapshot-id imaginarium-assets-v1 \
     --blender-bin /Applications/Blender.app/Contents/MacOS/Blender \
     --max-workers 3

For a separately managed OpenAI-compatible endpoint, use the explicit runtime
configuration form below instead.

.. code-block:: bash

   PYTHONPATH=src .venv/bin/python \
     scripts/run_non_rectangular_resilient_evaluation.py \
     --model-root gpt-5.6-sol=/abs/generation/gpt-5.6-sol \
     --model-root claude-opus-5-aihub=/abs/generation/claude-opus-5-aihub \
     --output-root Support/artifacts/outputs/non_rectangular_evaluation/my-run \
     --asset-csv /abs/Support/Assets/imaginarium_asset_info.csv \
     --asset-root /abs/Support/Assets/imaginarium_assets \
     --catalog-snapshot-id imaginarium-assets-v1 \
     --blender-bin /Applications/Blender.app/Contents/MacOS/Blender \
     --runtime-config configs/evaluation/non_rectangular_resilient_runtime.example.json \
     --max-workers 3

Use ``--resume`` only with the same output root, generation hashes, catalog,
Blender binary, runtime-config hash, and retry/concurrency settings.  An
ambiguous interrupted attempt fails closed.  ``--recover-interrupted`` is the
explicit safe-recovery switch; it opens a new immutable attempt and preserves
the earlier partial tree.

State machine and retry boundaries
----------------------------------

The fixed state order is generation preflight, room materialization,
independent materialization inspection, metric evaluation, room report, and
scene aggregation.  Functional precedes Semantic Placement.

* Every logical model call has one initial request plus exactly five transport
  retries, with the runtime-config backoff (30 seconds in the example).
* A room failure never blocks later rooms.  Fresh room attempts occur only
  after the complete initial pass, in global retry sweeps.
* Materialization and evaluator attempts have separate bounded budgets
  (three each by default).
* Blender crash/timeout/startup and filesystem interruption are retryable.
  Schema, polygon geometry, room assignment, asset identity, transform, and
  hash failures are non-retryable.
* Bounded local-camera exhaustion is finalized inside the explicit nonrect
  evaluator: retained/global plus geometry forced-choice comes first, followed
  by geometry-only binary adjudication and an audited deterministic fallback.
  API/transport, contract, identity, and corrupt-artifact failures remain
  evaluator failures and retain the existing retry taxonomy.

Materialization call graph
--------------------------

``run_resilient_nonrect_campaign`` calls strict generation preflight, projects
one authoritative ``RoomEvaluationUnit``, resolves every selected asset through
``FrozenCatalog``, writes a nonrect catalog plan, starts the dedicated Blender
build-only worker, and then reopens the room blend with a separate read-only
inspector.  The inspector reuses the canonical rigid-instance, material,
external-reference, hidden-state, and provenance checks while applying the
nonrect architecture allowlist for the exact polygon floor and ordered wall
segments.  The camera provider receives that room-only blend and the projected
polygon metadata.  Final aggregation calls the public API route with cached,
hash-verified room reports.

Room materialization artifacts
------------------------------

Each immutable successful attempt contains ``canonical_room_scene.json``,
``materialization_plan.json``, ``asset_resolution_manifest.json``,
``architecture_manifest.json``, ``source_identity.json``,
``prepared/room_evaluation.blend``, ``inspection_report.json``,
``materialization_manifest.json``, and the last-written ``complete.json``.
Every cache hit rehashes the full inventory and checks the source/config
identity before reuse.

Coordinator output
------------------

The output root contains ``run_manifest.json``, ``events.jsonl``,
``current_state.json``, ``provider_model_totals.json``, model/scene/room
summaries, per-attempt diagnostics, final scene reports, and
``terminal_manifest.json``.  Coordinator files contain no credentials,
endpoints, headers, prompt bodies, raw responses, or reasoning text.

For a no-API integration smoke run, replace ``--runtime-config`` with
``--mock-no-api``.  This still performs strict catalog binding and the complete
room materialization/evaluation state machine, using deterministic mock blend
and camera/evaluator artifacts.
