Evaluation Query and Relation Candidate Graphs
==============================================

These graphs are optional research and audit projections. They do not run
inside the evaluation Controller and have no authority over grouping, camera
selection, evidence budgets, rendering, EvidenceGate, or Judge decisions.

Relation candidate graph
------------------------

``RelationCandidateGraph`` records relations that may deserve evidence. It is
not a scene ground-truth graph. Each candidate retains one or more independent
source records:

* ``deterministic_geometry``: measured geometry or detector output;
* ``affordance_prior``: a versioned benchmark-owned candidate prior;
* ``vlm_hypothesis``: a non-judging discovery result.

Sources are never collapsed into one untraceable claim. A VLM hypothesis cannot
replace or edit a deterministic source record. Trusted grouping determines
whether a relation is within-group or cross-group; source-provided scope labels
are not authoritative.

.. code-block:: python

   from benchmark.visual_judge.graphs import build_relation_candidate_graph

   relations = build_relation_candidate_graph(
       case_id="N021",
       object_ids=("sofa", "television"),
       groups=(
           {
               "group_id": "group_001",
               "object_ids": ["sofa", "television"],
           },
       ),
       deterministic_candidates=(
           {
               "relation_type": "functional_correspondence",
               "relation_family": "functional",
               "target_ids": ["sofa", "television"],
               "observation_kinds": ["mutual_orientation"],
               "observation_goal": "Observe both ordinary-use sides.",
               "producer": "relative_orientation_detector",
               "method_version": "detector_v1",
               "source_ref": "event:001",
               "metadata": {},
           },
       ),
       functional_discovery=discovery_result,
   )

The lifecycle is audit-only:

.. code-block:: text

   candidate
   -> evidence_requested
   -> evidence_acquired
   -> adjudicated

The adjudicated state stores only an external ``decision_ref``; the relation
graph never generates a verdict.

Evaluation query graph
----------------------

``EvaluationQueryGraph`` projects one metric-scoped evaluation episode from the
contracts already used by the repository:

.. code-block:: text

   JudgeRequest
   + optional RelationCandidateGraph
   + optional Controller audit trace
   + optional JudgeResult
   -> post-hoc EvaluationQueryGraph

It connects:

* case, metric, claim and immutable scope;
* target object identities;
* relation candidates and their provenance;
* evidence artifacts;
* structured evidence requests;
* camera acquisition episodes and selection records;
* Judge calls and an externally produced final decision.

.. code-block:: python

   from benchmark.visual_judge.graphs import build_evaluation_query_graph

   query_graph = build_evaluation_query_graph(
       judge_request=judge_request,
       judge_result=judge_result,
       audit=controller_audit,
       relation_candidates=relations,
   )
   payload = query_graph.to_dict()

Projection is deterministic and read-only. The serialized form always records
``decision_authority = "none"`` and
``projection_mode = "posthoc_read_only"``.

Graph construction failure cannot change an evaluation result because the
Controller does not depend on this package. A runner may create the graph after
evaluation and store it beside the normal audit manifest.

Runner and exporter integration
-------------------------------

The camera-cal scene runner keeps graph generation disabled by default. Enable
the post-hoc export explicitly:

.. code-block:: console

   python scripts/run_camera_cal_scene_level.py \
     --output-root Support/artifacts/outputs/camera_cal_scene_level/example \
     --export-audit-graphs

The normal reports and verdicts are written first. The optional artifacts then
appear under ``cases/<case_id>/audit_graphs/``:

* ``relation_candidate_graph.json``;
* ``evaluation_query_graphs/*.json``;
* ``manifest.json``.

Controller audits retain the immutable ``JudgeRequest`` snapshot used for the
call. JSON-screen audits are persisted alongside the existing global and
group-local audits, so the exporter reads completed reports rather than
intercepting or replaying evaluation.

A completed compatible run can also be exported separately:

.. code-block:: console

   python scripts/export_evaluation_query_graphs.py \
     --run-root Support/artifacts/outputs/camera_cal_scene_level/example

Older reports that do not contain ``judge_request`` snapshots fail closed with
an export failure manifest. They are never reconstructed by guessing missing
request data. The local evidence viewer displays graph counts and provenance
only when this optional manifest is present.
