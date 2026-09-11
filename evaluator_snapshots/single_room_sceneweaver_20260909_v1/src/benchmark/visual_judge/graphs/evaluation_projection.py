"""Project current Judge and Controller audit artifacts into a query graph."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from benchmark.visual_judge.graphs._validation import (
    canonical_json,
    json_object,
    stable_id,
)
from benchmark.visual_judge.graphs.evaluation import (
    EVALUATION_QUERY_GRAPH_VERSION,
    EvaluationQueryGraph,
    QueryGraphEdge,
    QueryGraphNode,
)
from benchmark.visual_judge.graphs.relations import RelationCandidateGraph
from benchmark.visual_judge.interfaces.judge import (
    EvidenceRequest,
    JudgeRequest,
    JudgeResult,
)


_RELATION_FAMILIES_BY_METRIC = {
    "scale_consistency": frozenset({"geometric"}),
    "object_pairing_consistency": frozenset(
        {"affordance", "functional"}
    ),
    "style_consistency": frozenset(),
    "functional_consistency": frozenset(
        {
            "geometric",
            "functional",
            "affordance",
            "architectural",
            "circulation",
        }
    ),
    "semantic_placement_consistency": frozenset(
        {
            "geometric",
            "functional",
            "affordance",
            "architectural",
            "circulation",
        }
    ),
}


def build_evaluation_query_graph(
    *,
    judge_request: JudgeRequest | Mapping[str, Any],
    judge_result: JudgeResult | Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
    relation_candidates: RelationCandidateGraph
    | Mapping[str, Any]
    | None = None,
    case_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    workflow_context: Mapping[str, Any] | None = None,
) -> EvaluationQueryGraph:
    """Build a deterministic post-hoc projection with no runtime authority."""

    request = JudgeRequest.from_value(judge_request)
    result = (
        JudgeResult.from_value(judge_result)
        if judge_result is not None
        else None
    )
    audit_value = json_object(audit, "evaluation audit")
    workflow_value = json_object(
        workflow_context,
        "evaluation workflow context",
    )
    relations = (
        RelationCandidateGraph.from_value(relation_candidates)
        if relation_candidates is not None
        else None
    )
    case = _case_id(case_id, request, audit_value)
    if relations is not None and relations.case_id != case:
        raise ValueError("relation candidate graph case_id does not match")
    _validate_request_scope(request, relations)
    graph_id = stable_id(
        "evaluation_query",
        {
            "case_id": case,
            "metric": request.metric,
            "task": request.task,
            "claim": request.claim_or_event,
            "typed_obligation_ids": _typed_obligation_ids(request),
        },
    )
    builder = _Builder(graph_id=graph_id)
    evaluation = builder.node(
        "evaluation",
        f"{case}:{request.metric}",
        {"case_id": case, "task": request.task, "metric": request.metric},
        node_id=f"evaluation:{graph_id.split(':', 1)[1]}",
        provenance={"projection": EVALUATION_QUERY_GRAPH_VERSION},
    )
    metric = builder.node(
        "metric",
        request.metric,
        {"metric": request.metric},
        node_id=stable_id("metric", request.metric),
    )
    claim = builder.node(
        "claim",
        _claim_label(request),
        {
            "claim_or_event": request.claim_or_event,
            "rubric_ref": _rubric_ref(request.rubric),
        },
        node_id=stable_id(
            "claim",
            {"case_id": case, "metric": request.metric,
             "claim": request.claim_or_event},
        ),
        provenance={"source": "judge_request"},
    )
    builder.edge("contains_metric", evaluation, metric)
    builder.edge("contains_claim", evaluation, claim)
    builder.edge("under_metric", claim, metric)

    known = _known_objects(request, relations)
    targets = _target_ids(request)
    unknown = {item for item in targets if item != "scene" and known
               and item not in known}
    if unknown:
        raise ValueError(
            f"evaluation query references unknown object IDs: {sorted(unknown)}"
        )
    object_nodes: dict[str, str] = {}

    def object_node(object_id: str) -> str:
        if object_id == "scene":
            raise ValueError("'scene' is a scope, not an object")
        if known and object_id not in known:
            raise ValueError(f"unknown query object ID {object_id!r}")
        if object_id not in object_nodes:
            object_nodes[object_id] = builder.node(
                "object",
                object_id,
                {"object_id": object_id},
                node_id=stable_id("object", object_id),
                provenance={"source": "trusted_scene_identity"},
            )
        return object_nodes[object_id]

    for target in targets:
        if target != "scene":
            builder.edge("targets", claim, object_node(target))
    scope = _scope_node(builder, request)
    builder.edge("scoped_to", claim, scope)

    evidence_nodes: dict[str, str] = {}

    def evidence_node(item: Any, source: str) -> str:
        ref, summary = _evidence(item)
        if ref not in evidence_nodes:
            evidence_nodes[ref] = builder.node(
                "evidence_artifact",
                Path(ref).name or ref,
                summary,
                node_id=stable_id("evidence", ref),
                provenance={"source": source},
            )
        return evidence_nodes[ref]

    for item in request.visual_evidence:
        builder.edge("uses_evidence", claim, evidence_node(item, "judge_request"))
    relation_projection: dict[str, Any] = {
        "candidates": {},
    }
    if relations is not None:
        relation_projection = _relations(
            builder,
            relations,
            claim,
            targets,
            object_node,
            evidence_node,
            request=request,
            metric=request.metric,
        )
    typed_projection = _project_typed_workflow_inputs(
        builder=builder,
        request=request,
        audit=audit_value,
        workflow=workflow_value,
        claim=claim,
        scope=scope,
        object_node=object_node,
        request_evidence_nodes=tuple(evidence_nodes.values()),
        known_object_ids=known,
        relation_projection=relation_projection,
    )

    trace = _trace(audit_value)
    evidence_window_projection = _evidence_window_projection(audit_value)
    assignments, episodes = _episodes(trace)
    episode_nodes = {
        index: builder.node(
            "acquisition_episode",
            f"evidence repair episode {index}",
            summary,
            node_id=stable_id(
                "acquisition_episode",
                {"graph_id": graph_id, "episode_index": index},
            ),
            provenance={"source": "controller_audit_trace"},
        )
        for index, summary in sorted(episodes.items())
    }
    for node_id in episode_nodes.values():
        builder.edge("contains_episode", evaluation, node_id)

    request_nodes: dict[str, str] = {}

    def evidence_request_node(raw: Any, source: str) -> str:
        structured = EvidenceRequest.from_value(raw)
        key = canonical_json(structured.to_dict())
        if key not in request_nodes:
            request_nodes[key] = builder.node(
                "evidence_request",
                structured.view_goal,
                structured.to_dict(),
                node_id=stable_id(
                    "evidence_request",
                    {"graph_id": graph_id, "request": structured.to_dict()},
                ),
                provenance={"source": source},
            )
            for target in structured.target_ids:
                if target != "scene":
                    builder.edge(
                        "targets", request_nodes[key], object_node(target)
                    )
            builder.edge("scoped_to", request_nodes[key], scope)
        return request_nodes[key]

    judge_calls: list[str] = []
    last_trace_result: JudgeResult | None = None
    for position, event in enumerate(trace):
        stage = str(event.get("stage") or "")
        episode_index = assignments.get(position)
        episode = episode_nodes.get(episode_index)
        if stage == "functional_evidence_readiness":
            review = builder.node(
                "evidence_readiness_review",
                (
                    "CameraSelector evidence review: "
                    f"{event.get('status') or 'unknown'}"
                ),
                {
                    "status": event.get("status"),
                    "source": event.get("source"),
                    "check_id": event.get("check_id"),
                    "evidence_round": event.get("evidence_round"),
                    "result": deepcopy(event.get("result") or {}),
                    "decision_authority": "none",
                },
                node_id=stable_id(
                    "evidence_readiness_review",
                    {"graph_id": graph_id, "position": position},
                ),
                provenance={
                    "source": "controller_audit_trace",
                    "decision_authority": "none",
                },
            )
            builder.edge("reviews_evidence_for", claim, review)
            for item in event.get("images_used") or []:
                builder.edge(
                    "reviews_evidence",
                    review,
                    evidence_node(item, "readiness_trace"),
                )
            if isinstance(event.get("evidence_request"), Mapping):
                builder.edge(
                    "requests_evidence",
                    review,
                    evidence_request_node(
                        event["evidence_request"],
                        "camera_selector_readiness_trace",
                    ),
                )
        if stage == "camera_selector" and episode:
            selection = builder.node(
                "camera_selection",
                f"{event.get('selection_stage') or 'camera'} selection",
                _selection_summary(event),
                node_id=stable_id(
                    "camera_selection",
                    {"graph_id": graph_id, "position": position},
                ),
                provenance={"source": "controller_audit_trace"},
            )
            builder.edge("contains_selection", episode, selection)
        if stage == "render" and episode:
            for item in _render_evidence(event):
                builder.edge(
                    "produces_evidence",
                    episode,
                    evidence_node(item, "renderer_audit"),
                )
        if stage == "acquisition_planner" and event.get("evidence_request"):
            request_node = evidence_request_node(
                event["evidence_request"], "acquisition_planner_trace"
            )
            if episode:
                builder.edge("starts_episode", request_node, episode)
        if stage != "judge":
            continue
        raw_result = event.get("result")
        parsed = (
            JudgeResult.from_value(raw_result)
            if isinstance(raw_result, Mapping)
            else None
        )
        if parsed is not None:
            last_trace_result = parsed
        call = builder.node(
            "judge_call",
            f"Judge call {len(judge_calls) + 1}",
            _judge_summary(event, parsed),
            node_id=stable_id(
                "judge_call",
                {"graph_id": graph_id, "position": position},
            ),
            provenance={
                "source": "controller_audit_trace",
                "backend": parsed.backend if parsed else None,
            },
        )
        judge_calls.append(call)
        builder.edge("judged_by", claim, call)
        for item in event.get("images_used") or []:
            builder.edge(
                "uses_evidence", call, evidence_node(item, "judge_trace")
            )
        if parsed and parsed.evidence_request:
            builder.edge(
                "requests_evidence",
                call,
                evidence_request_node(
                    parsed.evidence_request, "judge_result"
                ),
            )

    if result is not None and last_trace_result is not None and (
        result.status != last_trace_result.status
        or result.confidence != last_trace_result.confidence
        or result.reason != last_trace_result.reason
    ):
        raise ValueError(
            "provided judge_result does not match the final audit Judge call"
        )
    effective_result = result or last_trace_result

    if not judge_calls and effective_result is not None:
        call = builder.node(
            "judge_call",
            "Judge call 1",
            {
                "status": effective_result.status,
                "confidence": effective_result.confidence,
                "evidence_round": None,
                "terminal_forced_choice": False,
            },
            node_id=stable_id(
                "judge_call", {"graph_id": graph_id, "source": "final"}
            ),
            provenance={
                "source": "provided_judge_result",
                "backend": effective_result.backend,
            },
        )
        judge_calls.append(call)
        builder.edge("judged_by", claim, call)
    if effective_result and effective_result.evidence_request and judge_calls:
        builder.edge(
            "requests_evidence",
            judge_calls[-1],
            evidence_request_node(
                effective_result.evidence_request,
                "provided_or_traced_judge_result",
            ),
        )
    decision: str | None = None
    if (
        effective_result
        and effective_result.status in {"valid", "invalid"}
        and judge_calls
    ):
        decision = builder.node(
            "decision",
            f"final {effective_result.status}",
            {
                "status": effective_result.status,
                "confidence": effective_result.confidence,
                "reason": effective_result.reason,
                "defect_count": len(effective_result.defects),
                "decision_ref": _decision_ref(effective_result),
            },
            node_id=stable_id(
                "decision",
                {"graph_id": graph_id, "status": effective_result.status,
                 "confidence": effective_result.confidence},
            ),
            provenance={
                "source": "judge_result",
                "backend": effective_result.backend,
            },
        )
        builder.edge("produces_decision", judge_calls[-1], decision)
    _project_typed_workflow_results(
        builder=builder,
        projection=typed_projection,
        judge_call=(judge_calls[-1] if judge_calls else None),
        decision=decision,
    )

    return EvaluationQueryGraph(
        graph_id=graph_id,
        case_id=case,
        metric=request.metric,
        nodes=builder.nodes(),
        edges=builder.edges(),
        metadata={
            **json_object(metadata, "evaluation query metadata"),
            "source_audit_schema_version": audit_value.get("schema_version"),
            "relation_candidate_graph_id": (
                stable_id("relation_graph", relations.to_dict())
                if relations else None
            ),
            "trace_event_count": len(trace),
            "typed_check_count": len(typed_projection["check_nodes"]),
            "check_result_count": len(typed_projection["result_specs"]),
            "ownership_event_count": len(
                typed_projection["ownership_nodes"]
            ),
            "evidence_window": evidence_window_projection,
            "functional_soft_evidence_contract": deepcopy(
                audit_value.get("functional_soft_evidence_contract")
            ),
        },
    )


def _evidence_window_projection(
    audit: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = audit.get("evidence_window")
    if not isinstance(value, Mapping):
        return None
    events = [
        event
        for event in value.get("events") or []
        if isinstance(event, Mapping)
    ]
    reused = list(
        dict.fromkeys(
            str(artifact_id)
            for event in events
            for artifact_id in event.get("reused_artifact_ids") or []
        )
    )
    evicted = list(
        dict.fromkeys(
            str(artifact_id)
            for event in events
            for artifact_id in event.get("evicted_artifact_ids") or []
        )
    )
    return {
        "schema_version": value.get("schema_version"),
        "policy": value.get("policy"),
        "group_id": value.get("group_id"),
        "check_id": value.get("check_id"),
        "max_active_images": value.get("max_active_images"),
        "fixed_artifact_ids": list(
            value.get("fixed_artifact_ids") or []
        ),
        "initial_artifact_ids": list(
            value.get("initial_artifact_ids") or []
        ),
        "final_artifact_ids": list(
            value.get("final_artifact_ids") or []
        ),
        "reused_artifact_ids": reused,
        "evicted_artifact_ids": evicted,
        "bank_reuse_event_count": sum(
            1 for event in events if event.get("reused_artifact_ids")
        ),
        "overflow_flush_count": sum(
            1
            for event in events
            if event.get("overflow_flush_applied") is True
        ),
        "physical_artifacts_deleted": False,
        "decision_authority": "none",
    }


def _typed_obligation_ids(request: JudgeRequest) -> list[str]:
    """Disambiguate isolated typed-check episodes within one claim scope."""

    result: list[str] = []
    for family, key in (
        ("functional", "required_functional_checks"),
        ("placement", "required_placement_checks"),
    ):
        raw_checks = request.context.get(key) or []
        if not isinstance(raw_checks, list):
            raise ValueError(f"JudgeRequest {key} must be a JSON list")
        for raw in raw_checks:
            if not isinstance(raw, Mapping):
                raise ValueError(f"JudgeRequest {key} must contain objects")
            check_id = str(raw.get("check_id") or "").strip()
            if not check_id:
                raise ValueError(f"JudgeRequest {key} requires check_id")
            result.append(f"{family}:{check_id}")
    return sorted(set(result))


class _Builder:
    def __init__(self, *, graph_id: str) -> None:
        self.graph_id = graph_id
        self._nodes: dict[str, QueryGraphNode] = {}
        self._edges: dict[str, QueryGraphEdge] = {}

    def node(
        self,
        kind: str,
        label: str,
        attributes: Mapping[str, Any] | None = None,
        *,
        node_id: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> str:
        node = QueryGraphNode(
            node_id=node_id,
            kind=kind,
            label=label,
            attributes=json_object(attributes, "query node attributes"),
            provenance=json_object(provenance, "query node provenance"),
        )
        prior = self._nodes.get(node_id)
        if prior is not None and prior != node:
            raise ValueError(f"query graph node collision {node_id!r}")
        self._nodes[node_id] = node
        return node_id

    def edge(
        self,
        kind: str,
        source: str,
        target: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        edge_id = stable_id(
            "edge",
            {"kind": kind, "source": source, "target": target,
             "attributes": attributes or {}},
        )
        edge = QueryGraphEdge(
            edge_id=edge_id,
            kind=kind,
            source_id=source,
            target_id=target,
            attributes=json_object(attributes, "query edge attributes"),
        )
        prior = self._edges.get(edge_id)
        if prior is not None and prior != edge:
            raise ValueError(f"query graph edge collision {edge_id!r}")
        self._edges[edge_id] = edge

    def nodes(self) -> tuple[QueryGraphNode, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda item: item.node_id))

    def edges(self) -> tuple[QueryGraphEdge, ...]:
        return tuple(sorted(self._edges.values(), key=lambda item: item.edge_id))


def _project_typed_workflow_inputs(
    *,
    builder: _Builder,
    request: JudgeRequest,
    audit: Mapping[str, Any],
    workflow: Mapping[str, Any],
    claim: str,
    scope: str,
    object_node: Any,
    request_evidence_nodes: tuple[str, ...],
    known_object_ids: set[str],
    relation_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Project typed obligations and ownership without granting authority."""

    if not known_object_ids and (
        request.context.get("required_functional_checks")
        or request.context.get("required_placement_checks")
        or workflow
    ):
        raise ValueError(
            "typed workflow projection requires trusted scene object IDs"
        )

    requested_ids: set[str] = set()
    deferred_ids: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {}

    def add_check(
        raw: Any,
        *,
        family: str,
        source: str,
        requested: bool = False,
        deferred: bool = False,
    ) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source} checks must contain JSON objects")
        check = deepcopy(dict(raw))
        check_id = str(check.get("check_id") or "").strip()
        check_type = str(
            check.get("check_type")
            or check.get("predicate")
            or ""
        ).strip()
        if not check_id or not check_type:
            raise ValueError(
                f"{source} check requires check_id and check_type"
            )
        target_ids = _workflow_ids(
            check.get("target_ids")
            or (
                [check.get("subject_id")]
                if check.get("subject_id")
                else []
            ),
            known=known_object_ids,
            label=f"{source} check target_ids",
        )
        context_ids = _workflow_ids(
            check.get("context_ids") or [],
            known=known_object_ids,
            label=f"{source} check context_ids",
        )
        subject_id = str(check.get("subject_id") or "").strip() or None
        if subject_id is not None and (
            subject_id not in known_object_ids
            or target_ids != [subject_id]
        ):
            raise ValueError(
                f"{source} placement check has an invalid subject identity"
            )
        core = {
            "family": family,
            "check_type": check_type,
            "subject_id": subject_id,
            "target_ids": target_ids,
            "context_ids": context_ids,
            "owner_stage": str(
                check.get("owner_stage") or ""
            ).strip(),
            "owning_group_id": (
                str(check.get("owning_group_id") or "").strip()
                or None
            ),
        }
        prior = records.get(check_id)
        if prior is not None and _check_core(prior) != core:
            raise ValueError(
                f"typed workflow check {check_id!r} has conflicting identity"
            )
        enriched = {
            **check,
            **core,
            "check_id": check_id,
        }
        if prior is None or _check_has_lifecycle(enriched):
            records[check_id] = enriched
        sources.setdefault(check_id, set()).add(source)
        if requested:
            requested_ids.add(check_id)
        if deferred:
            deferred_ids.add(check_id)

    for family, key in (
        ("functional", "required_functional_checks"),
        ("placement", "required_placement_checks"),
    ):
        raw_checks = request.context.get(key) or []
        if not isinstance(raw_checks, list):
            raise ValueError(f"JudgeRequest {key} must be a JSON list")
        for raw in raw_checks:
            add_check(
                raw,
                family=family,
                source=f"judge_request.{key}",
                requested=True,
            )

    raw_deferred_checks = request.context.get(
        "deferred_placement_checks"
    ) or []
    if not isinstance(raw_deferred_checks, list):
        raise ValueError(
            "JudgeRequest deferred_placement_checks must be a JSON list"
        )
    for raw in raw_deferred_checks:
        add_check(
            raw,
            family="placement",
            source="judge_request.deferred_placement_checks",
            deferred=True,
        )

    trace = audit.get("trace") or []
    if not isinstance(trace, list):
        raise ValueError("evaluation audit trace must be a JSON list")
    for event in trace:
        if not isinstance(event, Mapping):
            raise ValueError(
                "evaluation audit trace must contain JSON objects"
            )
        if str(event.get("stage") or "") != (
            "placement_check_lifecycle"
        ):
            continue
        lifecycle_check = event.get("check")
        is_deferred = bool(
            isinstance(lifecycle_check, Mapping)
            and (
                lifecycle_check.get("handoff_status")
                == "deferred_to_group_local"
            )
        )
        add_check(
            lifecycle_check,
            family="placement",
            source="controller_audit_trace",
            requested=not is_deferred,
            deferred=is_deferred,
        )

    workflow_ledgers = (
        ("functional", "functional_check_ledger"),
        ("placement", "placement_check_ledger"),
    )
    phase_ref = _workflow_phase_ref(request)
    for family, key in workflow_ledgers:
        ledger = workflow.get(key)
        if ledger is None:
            continue
        if not isinstance(ledger, Mapping):
            raise ValueError(f"workflow {key} must be a JSON object")
        raw_checks = ledger.get("checks")
        if not isinstance(raw_checks, list):
            raise ValueError(f"workflow {key} requires a checks list")
        for raw in raw_checks:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"workflow {key} checks must contain JSON objects"
                )
            check_id = str(raw.get("check_id") or "").strip()
            result_ref = str(
                raw.get("judge_result_ref") or ""
            ).strip()
            if (
                check_id not in requested_ids
                and not _phase_ref_matches(result_ref, phase_ref)
            ):
                continue
            add_check(
                raw,
                family=family,
                source=f"scene_report.{key}",
                requested=True,
            )

    ownership_records: dict[str, dict[str, Any]] = {}
    ownership_sources: dict[str, set[str]] = {}
    for ledger, source in (
        (
            request.context.get("functional_ownership_ledger"),
            "judge_request.functional_ownership_ledger",
        ),
        (
            workflow.get("functional_ownership_ledger"),
            "scene_report.functional_ownership_ledger",
        ),
    ):
        if ledger is None:
            continue
        if not isinstance(ledger, Mapping):
            raise ValueError(
                "functional ownership ledger must be a JSON object"
            )
        events = ledger.get("events")
        if not isinstance(events, list):
            raise ValueError(
                "functional ownership ledger requires an events list"
            )
        for raw in events:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    "functional ownership events must be JSON objects"
                )
            event = deepcopy(dict(raw))
            event_id = str(event.get("event_id") or "").strip()
            if not event_id:
                raise ValueError(
                    "functional ownership event requires event_id"
                )
            for field in (
                "affected_object_ids",
                "causal_object_ids",
                "scoring_target_ids",
            ):
                event[field] = _workflow_ids(
                    event.get(field) or [],
                    known=known_object_ids,
                    label=f"ownership event {field}",
                )
                if not event[field]:
                    raise ValueError(
                        f"ownership event {event_id!r} requires {field}"
                    )
            if not str(event.get("decision_ref") or "").strip():
                raise ValueError(
                    f"ownership event {event_id!r} requires decision_ref"
                )
            prior = ownership_records.get(event_id)
            if prior is not None and _ownership_core(prior) != (
                _ownership_core(event)
            ):
                raise ValueError(
                    f"ownership event {event_id!r} has conflicting identity"
                )
            ownership_records[event_id] = event
            ownership_sources.setdefault(event_id, set()).add(source)

    ownership_nodes: dict[str, str] = {}
    for event_id, event in sorted(ownership_records.items()):
        node = builder.node(
            "ownership_event",
            f"Functional ownership: {event_id}",
            event,
            node_id=stable_id("ownership_event", event_id),
            provenance={
                "sources": sorted(ownership_sources[event_id]),
                "decision_authority": "none",
            },
        )
        ownership_nodes[event_id] = node
        for object_id in event["affected_object_ids"]:
            builder.edge(
                "ownership_affects",
                node,
                object_node(object_id),
            )
        for object_id in event["causal_object_ids"]:
            builder.edge(
                "ownership_caused_by",
                node,
                object_node(object_id),
            )
        for object_id in event["scoring_target_ids"]:
            builder.edge(
                "ownership_scored_to",
                node,
                object_node(object_id),
            )

    check_nodes: dict[str, str] = {}
    result_specs: list[dict[str, Any]] = []
    for check_id, check in sorted(records.items()):
        attributes = {
            key: deepcopy(value)
            for key, value in check.items()
            if key != "result_row"
        }
        node = builder.node(
            "typed_check",
            f"{check['family']}:{check['check_type']}",
            attributes,
            node_id=stable_id("typed_check", check_id),
            provenance={
                "sources": sorted(sources[check_id]),
                "decision_authority": "none",
            },
        )
        check_nodes[check_id] = node
        for relation_node in _relation_nodes_for_check(
            check,
            relation_projection=relation_projection,
        ):
            builder.edge("routes_to_check", relation_node, node)
        builder.edge(
            "defers_check" if check_id in deferred_ids else "requires_check",
            claim,
            node,
        )
        builder.edge("scoped_to", node, scope)
        for object_id in check["target_ids"]:
            builder.edge(
                "check_targets",
                node,
                object_node(object_id),
            )
        for object_id in check["context_ids"]:
            builder.edge(
                "check_context",
                node,
                object_node(object_id),
            )
        if check_id not in deferred_ids:
            for evidence in request_evidence_nodes:
                builder.edge("check_uses_evidence", node, evidence)
        result_spec = _typed_check_result_spec(check)
        if result_spec is not None:
            result_specs.append(result_spec)

    return {
        "check_nodes": check_nodes,
        "result_specs": result_specs,
        "ownership_nodes": ownership_nodes,
    }


def _project_typed_workflow_results(
    *,
    builder: _Builder,
    projection: Mapping[str, Any],
    judge_call: str | None,
    decision: str | None,
) -> None:
    if judge_call is None and projection["result_specs"]:
        raise ValueError(
            "typed check results require a corresponding Judge call"
        )
    for spec in projection["result_specs"]:
        check_id = str(spec["check_id"])
        check_node = projection["check_nodes"].get(check_id)
        if check_node is None:
            raise ValueError(
                f"typed result references unknown check {check_id!r}"
            )
        result_node = builder.node(
            "check_result",
            f"{check_id}: {spec['conclusion']}",
            spec,
            node_id=stable_id(
                "check_result",
                {
                    "check_id": check_id,
                    "judge_result_ref": spec.get("judge_result_ref"),
                    "conclusion": spec["conclusion"],
                },
            ),
            provenance={
                "source": "validated_check_ledger",
                "decision_authority": "none",
            },
        )
        builder.edge(
            "produces_check_result",
            str(judge_call),
            result_node,
        )
        builder.edge("resolves_check", result_node, check_node)
        function_event_ref = str(
            spec.get("function_event_ref") or ""
        ).strip()
        if function_event_ref:
            ownership_node = projection["ownership_nodes"].get(
                function_event_ref
            )
            if ownership_node is None:
                raise ValueError(
                    "function-owned placement result references an unknown "
                    f"ownership event {function_event_ref!r}"
                )
            builder.edge(
                "excluded_by_ownership",
                result_node,
                ownership_node,
            )
        if decision is not None:
            builder.edge("supports_decision", result_node, decision)


def _typed_check_result_spec(
    check: Mapping[str, Any],
) -> dict[str, Any] | None:
    row = (
        deepcopy(dict(check["result_row"]))
        if isinstance(check.get("result_row"), Mapping)
        else {}
    )
    conclusion = str(
        row.get("conclusion")
        or check.get("check_conclusion")
        or ""
    ).strip()
    if not conclusion:
        return None
    if conclusion not in {
        "valid",
        "invalid",
        "excluded_function_owned",
        "unresolved",
    }:
        raise ValueError(
            f"typed check {check.get('check_id')!r} has invalid conclusion"
        )
    result = {
        "check_id": str(check["check_id"]),
        "family": str(check["family"]),
        "check_type": str(check["check_type"]),
        "conclusion": conclusion,
        "observation_status": (
            row.get("observation_status")
            or check.get("observation_status")
        ),
        "reason": row.get("reason"),
        "judge_result_ref": check.get("judge_result_ref"),
        "function_event_ref": (
            row.get("function_event_ref")
            or check.get("function_event_ref")
        ),
        "same_physical_event": row.get("same_physical_event"),
        "grounded": check.get("grounded"),
        "obligation_lifecycle": deepcopy(
            check.get("obligation_lifecycle") or []
        ),
    }
    return {
        key: deepcopy(value)
        for key, value in result.items()
        if value is not None
    }


def _workflow_ids(
    value: Any,
    *,
    known: set[str],
    label: str,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a JSON list")
    result = [str(item).strip() for item in value]
    if (
        any(not item for item in result)
        or len(result) != len(set(result))
        or any(item not in known for item in result)
    ):
        raise ValueError(f"{label} contains invalid object identities")
    return sorted(result)


def _check_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value.get(key))
        for key in (
            "family",
            "check_type",
            "subject_id",
            "target_ids",
            "context_ids",
            "owner_stage",
            "owning_group_id",
        )
    }


def _ownership_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value.get(key))
        for key in (
            "event_id",
            "affected_object_ids",
            "cause_kind",
            "causal_object_ids",
            "scoring_target_ids",
            "check_refs",
            "decision_ref",
        )
    }


def _check_has_lifecycle(value: Mapping[str, Any]) -> bool:
    return any(
        value.get(key) is not None
        for key in (
            "check_conclusion",
            "result_row",
            "judge_result_ref",
            "observation_status",
        )
    )


def _workflow_phase_ref(request: JudgeRequest) -> str:
    phase = str(
        request.context.get("evidence_phase") or ""
    ).strip()
    if phase == "group_local_review":
        group_scope = request.context.get("group_scope")
        group_id = (
            str(group_scope.get("group_id") or "").strip()
            if isinstance(group_scope, Mapping)
            else ""
        )
        return f"group_local_review:{group_id}" if group_id else phase
    if phase == "target_local_confirmation":
        target_scope = request.context.get("target_scope")
        target_id = (
            str(target_scope.get("target_id") or "").strip()
            if isinstance(target_scope, Mapping)
            else ""
        )
        return (
            f"target_local_confirmation:{target_id}"
            if target_id
            else phase
        )
    return phase


def _phase_ref_matches(result_ref: str, phase_ref: str) -> bool:
    if not result_ref or not phase_ref:
        return False
    if result_ref == phase_ref:
        return True
    return (
        phase_ref == "cross_group_relation_review"
        and result_ref.startswith("cross_group_relation_review:")
    )


def _relations(
    builder: _Builder,
    graph: RelationCandidateGraph,
    claim: str,
    targets: tuple[str, ...],
    object_node: Any,
    evidence_node: Any,
    *,
    request: JudgeRequest,
    metric: str,
) -> dict[str, Any]:
    global_query = not targets or "scene" in targets
    allowed_families = _RELATION_FAMILIES_BY_METRIC.get(
        metric,
        frozenset(),
    )
    projected: dict[str, dict[str, Any]] = {}
    for item in graph.candidates:
        if item.relation_family not in allowed_families:
            continue
        if not _relation_relevant_to_request(
            item,
            request=request,
            targets=targets,
            global_query=global_query,
        ):
            continue
        node = builder.node(
            "relation_candidate",
            f"{item.relation_type}:{','.join(item.target_ids)}",
            {
                "candidate_ref": item.candidate_id,
                "relation_type": item.relation_type,
                "relation_family": item.relation_family,
                "target_ids": list(item.target_ids),
                "observation_kinds": list(item.observation_kinds),
                "observation_goals": list(item.observation_goals),
                "scope": item.scope,
                "group_ids": list(item.group_ids),
                "state": item.state,
                "evidence_refs": list(item.evidence_refs),
                "decision_ref": item.decision_ref,
                "direct_claim_match": bool(
                    targets
                    and "scene" not in targets
                    and set(item.target_ids) == set(targets)
                ),
                "source_kinds": sorted(
                    {source.source_kind for source in item.sources}
                ),
            },
            node_id=stable_id("relation_candidate", item.candidate_id),
            provenance={
                "source": "relation_candidate_graph",
                "sources": [source.to_dict() for source in item.sources],
            },
        )
        builder.edge("examines_relation", claim, node)
        builder.edge(
            "scoped_to",
            node,
            _relation_scope_node(builder, graph, item),
        )
        for target in item.target_ids:
            builder.edge("relates", node, object_node(target))
        for evidence_ref in item.evidence_refs:
            builder.edge(
                "relation_has_evidence",
                node,
                evidence_node(
                    {
                        "path": evidence_ref,
                        "role": "relation_candidate_evidence",
                    },
                    "relation_candidate_graph",
                ),
            )
        projected[item.candidate_id] = {
            "node_id": node,
            "relation_type": item.relation_type,
            "target_ids": tuple(sorted(item.target_ids)),
            "scope": item.scope,
            "source_refs": tuple(
                sorted(source.source_ref for source in item.sources)
            ),
            "direct_claim_match": bool(
                targets
                and "scene" not in targets
                and set(item.target_ids) == set(targets)
            ),
        }
    return {"candidates": projected}


def _relation_relevant_to_request(
    candidate: Any,
    *,
    request: JudgeRequest,
    targets: tuple[str, ...],
    global_query: bool,
) -> bool:
    """Keep relation nodes aligned with the actual Judge responsibility."""

    phase = str(request.context.get("evidence_phase") or "").strip()
    candidate_targets = set(candidate.target_ids)
    request_targets = {item for item in targets if item != "scene"}
    if request.metric == "functional_consistency":
        required = request.context.get("required_functional_checks") or []
        required_types = {
            str(
                check.get("predicate")
                or check.get("check_type")
                or ""
            )
            for check in required
            if isinstance(check, Mapping)
        }
        if required_types and candidate.relation_type not in required_types:
            return False
        if phase == "global_discovery":
            # The global Functional Judge explicitly does not own discovered
            # cross-object relation claims.
            return False
        if phase == "cross_group_relation_review":
            return (
                candidate.scope == "cross_group"
                and candidate_targets == request_targets
            )
        if phase == "group_local_review":
            return (
                candidate.scope == "within_group"
                and bool(request_targets)
                and candidate_targets <= request_targets
            )
    if global_query:
        return True
    return bool(request_targets.intersection(candidate_targets))


def _relation_nodes_for_check(
    check: Mapping[str, Any],
    *,
    relation_projection: Mapping[str, Any],
) -> list[str]:
    """Resolve explicit discovery provenance to exact relation candidates."""

    if str(check.get("family") or "") != "functional":
        return []
    candidates = relation_projection.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        return []
    check_type = str(
        check.get("predicate") or check.get("check_type") or ""
    ).strip()
    target_ids = tuple(
        sorted(str(item) for item in check.get("target_ids") or [])
    )
    source_refs = {
        str(item)
        for field in ("source_discovery_ids", "routing_discovery_ids")
        for item in check.get(field) or []
        if str(item).strip()
    }
    exact: list[Mapping[str, Any]] = [
        candidate
        for candidate in candidates.values()
        if isinstance(candidate, Mapping)
        and candidate.get("relation_type") == check_type
        and tuple(candidate.get("target_ids") or ()) == target_ids
    ]
    if source_refs:
        referenced = [
            candidate
            for candidate in candidates.values()
            if isinstance(candidate, Mapping)
            and source_refs.intersection(candidate.get("source_refs") or ())
        ]
        exact = [
            candidate
            for candidate in exact
            if source_refs.intersection(candidate.get("source_refs") or ())
        ]
        if referenced and not exact:
            raise ValueError(
                "functional check discovery provenance conflicts with its "
                "relation candidate identity"
            )
    if len(exact) > 1:
        raise ValueError(
            "functional check maps to multiple relation candidates"
        )
    return [str(candidate["node_id"]) for candidate in exact]


def _relation_scope_node(
    builder: _Builder,
    graph: RelationCandidateGraph,
    candidate: Any,
) -> str:
    attributes = {
        "scope_type": candidate.scope,
        "group_ids": list(candidate.group_ids),
        "target_ids": list(candidate.target_ids),
    }
    if candidate.scope == "within_group":
        group_id = candidate.group_ids[0]
        attributes["group_id"] = group_id
        attributes["member_ids"] = sorted(
            object_id
            for object_id, trusted_group_id in graph.group_membership.items()
            if trusted_group_id == group_id
        )
    return builder.node(
        "scope",
        f"relation scope {candidate.scope}",
        attributes,
        node_id=stable_id(
            "scope",
            {
                "relation_candidate_id": candidate.candidate_id,
                "scope": candidate.scope,
                "group_ids": candidate.group_ids,
            },
        ),
        provenance={"source": "trusted_relation_candidate_graph"},
    )


def _scope_node(builder: _Builder, request: JudgeRequest) -> str:
    target = request.context.get("target_scope")
    if isinstance(target, Mapping) and str(
        target.get("target_id") or ""
    ).strip():
        target_id = str(target["target_id"])
        context_ids = [
            str(item) for item in target.get("context_ids") or []
        ]
        return builder.node(
            "scope",
            f"target scope {target_id}",
            {
                "scope_type": "target_centered_context",
                "scope_id": target.get("scope_id"),
                "target_id": target_id,
                "context_ids": context_ids,
                "framing_ids": list(target.get("framing_ids") or []),
                "focus_center": target.get("focus_center"),
                "extent": target.get("extent"),
                "require_global_anchor": bool(
                    target.get("require_global_anchor")
                ),
                "context_objects_are_defect_owners": False,
                "group_identity": None,
            },
            node_id=stable_id(
                "scope",
                {
                    "metric": request.metric,
                    "scope": "target_centered_context",
                    "target_id": target_id,
                    "context_ids": context_ids,
                },
            ),
            provenance={
                "source": "trusted_target_scope",
                "scope_version": target.get("scope_version"),
            },
        )
    raw = request.context.get("group_scope")
    if isinstance(raw, Mapping) and str(raw.get("group_id") or "").strip():
        group_id = str(raw["group_id"])
        return builder.node(
            "scope",
            f"group scope {group_id}",
            {
                "scope_type": "group",
                "group_id": group_id,
                "member_ids": list(raw.get("member_ids") or []),
                "focus_center": raw.get("focus_center"),
                "extent": raw.get("extent"),
            },
            node_id=stable_id(
                "scope",
                {"metric": request.metric, "group_id": group_id,
                 "member_ids": raw.get("member_ids") or []},
            ),
            provenance={
                "source": "trusted_group_scope",
                "grouping_policy_id": raw.get("grouping_policy_id"),
                "grouping_backend": raw.get("grouping_backend"),
            },
        )
    return builder.node(
        "scope",
        "scene scope",
        {"scope_type": "scene"},
        node_id=stable_id("scope", {"metric": request.metric, "scope": "scene"}),
        provenance={"source": "judge_request"},
    )


def _validate_request_scope(
    request: JudgeRequest,
    relations: RelationCandidateGraph | None,
) -> None:
    """Fail closed when a request claims an untrusted group partition."""

    _validate_target_scope(request, relations)
    if relations is None:
        return
    raw = request.context.get("group_scope")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError("judge request group_scope must be an object")
    group_id = str(raw.get("group_id") or "").strip()
    if not group_id:
        raise ValueError("judge request group_scope requires group_id")
    members = tuple(
        sorted(str(item).strip() for item in raw.get("member_ids") or [])
    )
    if not members or any(not item for item in members):
        raise ValueError("judge request group_scope requires member_ids")
    trusted_members = tuple(
        sorted(
            object_id
            for object_id, trusted_group_id in (
                relations.group_membership.items()
            )
            if trusted_group_id == group_id
        )
    )
    if not trusted_members:
        raise ValueError(
            f"judge request references unknown group_id {group_id!r}"
        )
    if members != trusted_members:
        raise ValueError(
            "judge request group_scope does not match the trusted partition"
        )


def _validate_target_scope(
    request: JudgeRequest,
    relations: RelationCandidateGraph | None,
) -> None:
    raw = request.context.get("target_scope")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError("judge request target_scope must be an object")
    if request.context.get("group_scope") is not None:
        raise ValueError(
            "judge request cannot claim group_scope and target_scope together"
        )
    target_id = str(raw.get("target_id") or "").strip()
    if not target_id:
        raise ValueError("judge request target_scope requires target_id")
    context_ids = tuple(
        str(item).strip() for item in raw.get("context_ids") or []
    )
    framing_ids = tuple(
        str(item).strip() for item in raw.get("framing_ids") or []
    )
    if (
        any(not item for item in context_ids)
        or len(set(context_ids)) != len(context_ids)
        or target_id in context_ids
    ):
        raise ValueError("judge request target_scope has invalid context_ids")
    if (
        not framing_ids
        or framing_ids[0] != target_id
        or framing_ids != (target_id, *context_ids)
    ):
        raise ValueError(
            "judge request target_scope framing_ids must be target then context"
        )
    if len(context_ids) > 3:
        raise ValueError(
            "judge request target_scope exceeds bounded context capacity"
        )
    if raw.get("group_identity") is not None:
        raise ValueError("judge request target_scope cannot claim group identity")
    if raw.get("context_objects_are_defect_owners") not in (None, False):
        raise ValueError(
            "judge request target_scope context objects must be non-owning"
        )
    allowed_targets = tuple(
        str(item).strip()
        for item in request.context.get("target_object_ids") or []
        if str(item).strip()
    )
    if allowed_targets and set(allowed_targets) != {target_id}:
        raise ValueError(
            "judge request target_scope attribution must remain target-only"
        )
    claimed_targets = {
        item for item in _target_ids(request) if item != "scene"
    }
    if claimed_targets != {target_id}:
        raise ValueError(
            "judge request target_scope claim must remain target-only"
        )
    known = _known_objects(request, relations)
    unknown = {
        item for item in framing_ids if known and item not in known
    }
    if unknown:
        raise ValueError(
            "judge request target_scope references unknown object IDs: "
            f"{sorted(unknown)}"
        )


def _case_id(
    explicit: str | None,
    request: JudgeRequest,
    audit: Mapping[str, Any],
) -> str:
    evaluation = audit.get("evaluation")
    candidates = (
        explicit,
        request.context.get("case_id"),
        request.context.get("scene_id"),
        request.scene_context.get("case_id"),
        request.scene_context.get("scene_id"),
        evaluation.get("case_id") if isinstance(evaluation, Mapping) else None,
    )
    for value in candidates:
        if str(value or "").strip():
            return str(value).strip()
    return stable_id(
        "case",
        {"task": request.task, "metric": request.metric,
         "scene_context": request.scene_context},
    )


def _known_objects(
    request: JudgeRequest,
    relations: RelationCandidateGraph | None,
) -> set[str]:
    result = set(relations.object_ids) if relations else set()
    for item in request.scene_context.get("objects") or []:
        if isinstance(item, Mapping):
            object_id = str(
                item.get("id") or item.get("object_id") or ""
            ).strip()
            if object_id:
                result.add(object_id)
    return result


def _target_ids(request: JudgeRequest) -> tuple[str, ...]:
    values: list[str] = []
    claim = request.claim_or_event
    for key in ("target_ids", "object_ids", "member_ids"):
        if isinstance(claim.get(key), (list, tuple)):
            values.extend(str(item).strip() for item in claim[key])
    for key in (
        "target_id", "object_id", "object_a_id", "object_b_id",
        "subject_id", "reference_id", "source_id",
    ):
        if isinstance(claim.get(key), (str, int)):
            values.append(str(claim[key]).strip())
    for key in ("target_object_ids", "object_ids", "target_ids"):
        raw = request.context.get(key)
        if isinstance(raw, (list, tuple)):
            values.extend(str(item).strip() for item in raw)
    scope = request.context.get("group_scope")
    if isinstance(scope, Mapping):
        values.extend(str(item).strip() for item in scope.get("member_ids") or [])
    target_scope = request.context.get("target_scope")
    if isinstance(target_scope, Mapping):
        values.append(str(target_scope.get("target_id") or "").strip())
    return tuple(dict.fromkeys(item for item in values if item))


def _trace(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = audit.get("trace")
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError("evaluation audit trace must be a list of objects")
    return [deepcopy(dict(item)) for item in value]


def _episodes(
    trace: list[dict[str, Any]],
) -> tuple[dict[int, int | None], dict[int, dict[str, Any]]]:
    assignments: dict[int, int | None] = {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    current: int | None = None
    for position, event in enumerate(trace):
        value = event.get("episode_index")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            current = value
        assignments[position] = current
        if current is not None:
            grouped.setdefault(current, []).append(event)
    summaries = {}
    for index, events in grouped.items():
        rounds = [
            item["evidence_round"] for item in events
            if isinstance(item.get("evidence_round"), int)
            and not isinstance(item.get("evidence_round"), bool)
        ]
        summaries[index] = {
            "episode_index": index,
            "stages": list(dict.fromkeys(
                str(item.get("stage") or "") for item in events
                if str(item.get("stage") or "")
            )),
            "selection_stages": list(dict.fromkeys(
                str(item.get("selection_stage") or "") for item in events
                if str(item.get("selection_stage") or "")
            )),
            "first_evidence_round": min(rounds) if rounds else None,
            "last_evidence_round": max(rounds) if rounds else None,
            "event_count": len(events),
        }
    return assignments, summaries


def _evidence(item: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(item, (str, Path)):
        ref = str(item)
        return ref, {"evidence_ref": ref}
    if not isinstance(item, Mapping):
        ref = stable_id("inline_evidence", repr(item))
        return ref, {"evidence_ref": ref, "representation": type(item).__name__}
    ref = next(
        (str(item[key]).strip() for key in (
            "path", "image_path", "artifact_path", "file_path", "ref"
        ) if item.get(key) is not None and str(item[key]).strip()),
        "",
    )
    if not ref:
        view_id = str(item.get("view_id") or item.get("id") or "").strip()
        ref = f"view:{view_id}" if view_id else stable_id("inline_evidence", item)
    summary = {
        "evidence_ref": ref,
        "view_id": item.get("view_id") or item.get("id"),
        "role": item.get("role") or item.get("evidence_role"),
        "representation": item.get("representation"),
        "evidence_round": item.get("evidence_round"),
        "camera_pose_id": item.get("camera_pose_id"),
    }
    return ref, {key: value for key, value in summary.items()
                 if value is not None}


def _render_evidence(event: Mapping[str, Any]) -> list[Any]:
    result = event.get("result")
    if isinstance(result, Mapping) and isinstance(
        result.get("visual_evidence"), (list, tuple)
    ):
        return list(result["visual_evidence"])
    return list(event.get("images_used") or [])


def _selection_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    result = result if isinstance(result, Mapping) else {}
    return {
        "selection_stage": event.get("selection_stage"),
        "evidence_round": event.get("evidence_round"),
        "outcome": event.get("outcome") or result.get("outcome"),
        "backend": (
            event.get("backend") or event.get("selector_backend")
            or result.get("backend")
        ),
        "selected_view_ids": list(
            event.get("selected_view_ids")
            or result.get("selected_view_ids") or []
        ),
        "selected_plan_id": (
            event.get("selected_plan_id") or result.get("selected_plan_id")
        ),
        "reason_codes": list(
            event.get("reason_codes") or result.get("reason_codes") or []
        ),
    }


def _judge_summary(
    event: Mapping[str, Any],
    result: JudgeResult | None,
) -> dict[str, Any]:
    return {
        "evidence_round": event.get("evidence_round"),
        "status": result.status if result else None,
        "confidence": result.confidence if result else None,
        "image_count": len(event.get("images_used") or []),
        "terminal_forced_choice": bool(event.get("terminal_forced_choice")),
    }


def _claim_label(request: JudgeRequest) -> str:
    for key in ("claim_id", "event_id", "id", "type", "relation_type"):
        if str(request.claim_or_event.get(key) or "").strip():
            return f"{request.metric}: {request.claim_or_event[key]}"
    return f"{request.metric} claim"


def _rubric_ref(value: Any) -> str | None:
    if value is None:
        return None
    return value[:500] if isinstance(value, str) else stable_id("rubric", value)


def _decision_ref(result: JudgeResult) -> str:
    value = str(result.provenance.get("decision_ref") or "").strip()
    return value or stable_id(
        "judge_decision",
        {
            "status": result.status,
            "confidence": result.confidence,
            "reason": result.reason,
            "defects": result.defects,
        },
    )
