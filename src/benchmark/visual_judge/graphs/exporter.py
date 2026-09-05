"""Opt-in, post-hoc export of non-decisional evaluation audit graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmark.visual_judge.graphs._validation import (
    canonical_json,
    stable_id,
)
from benchmark.visual_judge.graphs.evaluation_projection import (
    build_evaluation_query_graph,
)
from benchmark.visual_judge.graphs.relation_builders import (
    build_relation_candidate_graph,
    transition_relation_candidate,
)
from benchmark.visual_judge.interfaces import JudgeRequest


AUDIT_GRAPH_EXPORT_VERSION = "evaluation_audit_graph_export_v1"
_AUDIT_FIELD_SUFFIX = "camera_control_audit"


def export_case_audit_graphs(
    *,
    case_id: str,
    grouping_report: Mapping[str, Any],
    scene_quality_report: Mapping[str, Any],
    output_dir: str | Path,
    deterministic_candidates: Iterable[Mapping[str, Any]] = (),
    affordance_candidates: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project completed report artifacts into validated audit-only graphs.

    Construction is completed fully in memory before any graph is written.
    Invalid identity, partition, provenance, request, or edge data therefore
    produces only a failure manifest and never a partial authoritative graph.
    """

    target = Path(output_dir).expanduser().resolve()
    try:
        bundle = _build_bundle(
            case_id=case_id,
            grouping_report=grouping_report,
            scene_quality_report=scene_quality_report,
            deterministic_candidates=deterministic_candidates,
            affordance_candidates=affordance_candidates,
        )
    except Exception as exc:
        manifest = {
            "schema_version": AUDIT_GRAPH_EXPORT_VERSION,
            "case_id": str(case_id),
            "status": "failed",
            "decision_authority": "none",
            "projection_mode": "posthoc_read_only",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "relation_candidate_graph": None,
            "evaluation_query_graphs": [],
        }
        _atomic_write_json(target / "manifest.json", manifest)
        return manifest

    relation_graph = bundle["relation_graph"]
    query_graphs = bundle["query_graphs"]
    relation_path = target / "relation_candidate_graph.json"
    query_dir = target / "evaluation_query_graphs"
    graph_records: list[dict[str, Any]] = []
    _atomic_write_json(relation_path, relation_graph)
    for item in query_graphs:
        graph = item["graph"]
        digest = str(graph["graph_id"]).split(":", 1)[-1]
        metric = _safe_token(str(graph["metric"]))
        filename = f"{item['sequence']:04d}_{metric}_{digest}.json"
        graph_path = query_dir / filename
        _atomic_write_json(graph_path, graph)
        graph_records.append(
            {
                "sequence": item["sequence"],
                "graph_id": graph["graph_id"],
                "metric": graph["metric"],
                "source_path": item["source_path"],
                "source_paths": item["source_paths"],
                "path": str(graph_path.relative_to(target)),
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
                "node_kind_counts": _kind_counts(
                    graph["nodes"],
                ),
                "edge_kind_counts": _kind_counts(
                    graph["edges"],
                ),
            }
        )
    manifest = {
        "schema_version": AUDIT_GRAPH_EXPORT_VERSION,
        "case_id": str(case_id),
        "status": "complete",
        "decision_authority": "none",
        "projection_mode": "posthoc_read_only",
        "source_report_sha256": hashlib.sha256(
            canonical_json(scene_quality_report).encode("utf-8")
        ).hexdigest(),
        "relation_candidate_graph": {
            "path": str(relation_path.relative_to(target)),
            "candidate_count": len(relation_graph["candidates"]),
            "object_count": len(relation_graph["object_ids"]),
        },
        "evaluation_query_graphs": graph_records,
    }
    _atomic_write_json(target / "manifest.json", manifest)
    return manifest


def _build_bundle(
    *,
    case_id: str,
    grouping_report: Mapping[str, Any],
    scene_quality_report: Mapping[str, Any],
    deterministic_candidates: Iterable[Mapping[str, Any]],
    affordance_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = grouping_report.get("object_groups")
    if not isinstance(groups, list):
        raise ValueError("grouping report requires object_groups")
    object_ids = _object_ids(grouping_report, groups)
    functional_discovery = _functional_discovery(scene_quality_report)
    relation_graph = build_relation_candidate_graph(
        case_id=str(case_id),
        object_ids=object_ids,
        groups=groups,
        deterministic_candidates=deterministic_candidates,
        affordance_candidates=affordance_candidates,
        functional_discovery=functional_discovery,
        metadata={
            "purpose": "posthoc_evaluation_audit",
            "source": "completed_scene_quality_report",
            "non_decisional": True,
        },
    )

    audit_records: list[dict[str, Any]] = []
    audit_records_by_payload: dict[str, dict[str, Any]] = {}
    for source_path, record in _camera_control_audits(
        scene_quality_report
    ):
        audit = _audit_payload(record)
        payload_key = canonical_json(audit)
        prior_record = audit_records_by_payload.get(payload_key)
        if prior_record is not None:
            prior_record["source_paths"].append(source_path)
            continue
        audit_record = {
            "source_path": source_path,
            "source_paths": [source_path],
            "audit": audit,
        }
        audit_records_by_payload[payload_key] = audit_record
        audit_records.append(audit_record)

    episodes: list[dict[str, Any]] = []
    for sequence, audit_record in enumerate(audit_records, start=1):
        source_path = audit_record["source_path"]
        source_paths = tuple(audit_record["source_paths"])
        audit = audit_record["audit"]
        raw_request = audit.get("judge_request")
        if not isinstance(raw_request, Mapping):
            raise ValueError(
                f"{source_path} does not retain its JudgeRequest snapshot"
            )
        request = JudgeRequest.from_value(raw_request)
        audit_evaluation = audit.get("evaluation")
        if isinstance(audit_evaluation, Mapping):
            audit_metric = str(audit_evaluation.get("metric") or "").strip()
            if audit_metric and audit_metric != request.metric:
                raise ValueError(
                    f"{source_path} metric disagrees with JudgeRequest"
                )
        graph = build_evaluation_query_graph(
            judge_request=request,
            audit=audit,
            relation_candidates=relation_graph,
            case_id=str(case_id),
            workflow_context=_metric_workflow_context(
                scene_quality_report,
                request.metric,
            ),
            metadata={
                "source_path": source_path,
                "source_paths": source_paths,
                "source_kind": "persisted_controller_audit",
            },
        ).to_dict()
        episodes.append(
            {
                "sequence": sequence,
                "source_path": source_path,
                "source_paths": list(source_paths),
                "request": request,
                "audit": audit,
                "graph": graph,
            }
        )

    relation_graph = _project_relation_lifecycle(
        relation_graph,
        [item["graph"] for item in episodes],
    )
    query_graphs: list[dict[str, Any]] = []
    graph_ids: dict[str, dict[str, Any]] = {}
    for item in episodes:
        graph = build_evaluation_query_graph(
            judge_request=item["request"],
            audit=item["audit"],
            relation_candidates=relation_graph,
            case_id=str(case_id),
            workflow_context=_metric_workflow_context(
                scene_quality_report,
                item["request"].metric,
            ),
            metadata={
                "source_path": item["source_path"],
                "source_paths": item["source_paths"],
                "source_kind": "persisted_controller_audit",
            },
        ).to_dict()
        prior = graph_ids.get(graph["graph_id"])
        if prior is not None and prior != graph:
            raise ValueError(
                f"conflicting query projections share {graph['graph_id']}"
            )
        if prior is not None:
            continue
        graph_ids[graph["graph_id"]] = graph
        query_graphs.append(
            {
                "sequence": item["sequence"],
                "source_path": item["source_path"],
                "source_paths": item["source_paths"],
                "graph": graph,
            }
        )
    return {
        "relation_graph": relation_graph.to_dict(),
        "query_graphs": query_graphs,
    }


def _project_relation_lifecycle(
    relation_graph: Any,
    query_graphs: list[dict[str, Any]],
) -> Any:
    """Attach external workflow references without deriving a verdict."""

    records: dict[str, dict[str, set[str]]] = {}
    for graph in query_graphs:
        nodes = {
            str(node["node_id"]): node
            for node in graph.get("nodes") or []
            if isinstance(node, Mapping)
        }
        candidate_nodes = {
            str(node_id): str(
                (node.get("attributes") or {}).get("candidate_ref")
            )
            for node_id, node in nodes.items()
            if node.get("kind") == "relation_candidate"
            and isinstance(node.get("attributes"), Mapping)
            and (node.get("attributes") or {}).get("candidate_ref")
        }
        if not candidate_nodes:
            continue
        edges = [
            edge
            for edge in graph.get("edges") or []
            if isinstance(edge, Mapping)
        ]
        checks_by_candidate: dict[str, set[str]] = {
            node_id: set() for node_id in candidate_nodes
        }
        for edge in edges:
            if (
                edge.get("kind") == "routes_to_check"
                and str(edge.get("source_id")) in candidate_nodes
            ):
                checks_by_candidate[str(edge["source_id"])].add(
                    str(edge.get("target_id"))
                )
        results_by_check: dict[str, set[str]] = {}
        for edge in edges:
            if edge.get("kind") == "resolves_check":
                results_by_check.setdefault(
                    str(edge.get("target_id")),
                    set(),
                ).add(str(edge.get("source_id")))
        evidence_by_check: dict[str, set[str]] = {}
        for edge in edges:
            if edge.get("kind") == "check_uses_evidence":
                evidence_by_check.setdefault(
                    str(edge.get("source_id")),
                    set(),
                ).add(str(edge.get("target_id")))
        request_nodes = {
            str(node_id): node
            for node_id, node in nodes.items()
            if node.get("kind") == "evidence_request"
        }
        graph_decision_refs = {
            str((node.get("attributes") or {}).get("decision_ref"))
            for node in nodes.values()
            if node.get("kind") == "decision"
            and isinstance(node.get("attributes"), Mapping)
            and (node.get("attributes") or {}).get("decision_ref")
        }
        produced_evidence_ids = {
            str(edge.get("target_id"))
            for edge in edges
            if edge.get("kind") == "produces_evidence"
        }
        for candidate_node_id, candidate_id in candidate_nodes.items():
            linked_checks = checks_by_candidate.get(candidate_node_id) or set()
            direct_claim_match = bool(
                (nodes[candidate_node_id].get("attributes") or {}).get(
                    "direct_claim_match"
                )
            )
            if not linked_checks and not direct_claim_match:
                # Presence in a broad query is not a lifecycle transition.
                continue
            result_nodes_by_check = {
                check_id: results_by_check.get(check_id) or set()
                for check_id in linked_checks
            }
            resolved = (
                all(
                    result_nodes
                    and all(
                        str(
                            (
                                nodes.get(result_id, {}).get("attributes")
                                or {}
                            ).get("conclusion")
                            or ""
                        )
                        in {"valid", "invalid"}
                        for result_id in result_nodes
                    )
                    for result_nodes in result_nodes_by_check.values()
                )
                if linked_checks
                else bool(graph_decision_refs)
            )
            evidence_node_ids = (
                {
                    evidence_id
                    for check_id in linked_checks
                    for evidence_id in evidence_by_check.get(check_id) or set()
                }
                if linked_checks
                else set(produced_evidence_ids)
            )
            evidence_refs = {
                str(
                    (nodes[evidence_id].get("attributes") or {}).get(
                        "evidence_ref"
                    )
                )
                for evidence_id in evidence_node_ids
                if evidence_id in nodes
                and isinstance(nodes[evidence_id].get("attributes"), Mapping)
                and (nodes[evidence_id].get("attributes") or {}).get(
                    "evidence_ref"
                )
            }
            candidate_targets = set(
                (nodes[candidate_node_id].get("attributes") or {}).get(
                    "target_ids"
                )
                or []
            )
            request_refs = {
                request_id
                for request_id, request_node in request_nodes.items()
                if (
                    request_targets := set(
                        (request_node.get("attributes") or {}).get(
                            "target_ids"
                        )
                        or []
                    )
                )
                and request_targets <= candidate_targets
            }
            decision_refs = graph_decision_refs if resolved else set()
            record = records.setdefault(
                candidate_id,
                {
                    "evidence_request_refs": set(),
                    "evidence_refs": set(),
                    "decision_refs": set(),
                    "query_graph_ids": set(),
                },
            )
            record["evidence_request_refs"].update(request_refs)
            record["evidence_refs"].update(evidence_refs)
            record["decision_refs"].update(decision_refs)
            record["query_graph_ids"].add(str(graph["graph_id"]))

    projected = relation_graph
    for candidate in relation_graph.candidates:
        record = records.get(candidate.candidate_id)
        if not record:
            continue
        evidence_refs = sorted(record["evidence_refs"])
        decision_refs = sorted(record["decision_refs"])
        if decision_refs:
            decision_ref = (
                decision_refs[0]
                if len(decision_refs) == 1
                else stable_id("decision_set", decision_refs)
            )
            projected = transition_relation_candidate(
                projected,
                candidate_id=candidate.candidate_id,
                state="adjudicated",
                evidence_refs=evidence_refs,
                decision_ref=decision_ref,
            )
        elif evidence_refs:
            projected = transition_relation_candidate(
                projected,
                candidate_id=candidate.candidate_id,
                state="evidence_acquired",
                evidence_refs=evidence_refs,
            )
        elif record["evidence_request_refs"]:
            projected = transition_relation_candidate(
                projected,
                candidate_id=candidate.candidate_id,
                state="evidence_requested",
            )

    lifecycle = {
        candidate_id: {
            key: sorted(values)
            for key, values in sorted(record.items())
        }
        for candidate_id, record in sorted(records.items())
    }
    return replace(
        projected,
        metadata={
            **deepcopy(projected.metadata),
            "lifecycle_projection": lifecycle,
            "lifecycle_authority": "external_audit_only",
        },
    )


def _object_ids(
    grouping_report: Mapping[str, Any],
    groups: list[Any],
) -> tuple[str, ...]:
    catalog = grouping_report.get("object_catalog")
    if isinstance(catalog, list) and catalog:
        values = [
            str(item.get("object_id") or "").strip()
            for item in catalog
            if isinstance(item, Mapping)
        ]
    else:
        values = [
            str(object_id).strip()
            for group in groups
            if isinstance(group, Mapping)
            for object_id in group.get("object_ids") or []
        ]
    if not values or any(not value for value in values):
        raise ValueError("grouping report contains invalid object identities")
    if len(values) != len(set(values)):
        raise ValueError("grouping report object identities must be unique")
    return tuple(sorted(values))


def _functional_discovery(
    report: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    functional = metrics.get("functional_consistency")
    functional = functional if isinstance(functional, Mapping) else {}
    value = functional.get("functional_discovery")
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    prejudgement = functional.get("functional_prejudgement_evidence")
    if isinstance(prejudgement, Mapping):
        nested = prejudgement.get("functional_discovery")
        if isinstance(nested, Mapping):
            return deepcopy(dict(nested))
    return None


def _metric_workflow_context(
    report: Mapping[str, Any],
    metric: str,
) -> dict[str, Any]:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    record = metrics.get(metric)
    record = record if isinstance(record, Mapping) else {}
    result: dict[str, Any] = {}
    for key in (
        "functional_check_ledger",
        "functional_ownership_ledger",
        "placement_check_ledger",
        "placement_check_coverage",
        "cross_metric_ownership_audit",
    ):
        value = record.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"metric workflow field {key!r} must be an object"
                )
            result[key] = deepcopy(dict(value))
    return result


def _camera_control_audits(
    value: Any,
    *,
    path: str = "$",
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = value[key]
            child_path = f"{path}.{key}"
            if str(key).endswith(_AUDIT_FIELD_SUFFIX):
                if not isinstance(child, Mapping):
                    raise ValueError(f"{child_path} must be an object")
                result.append((child_path, deepcopy(dict(child))))
                continue
            result.extend(_camera_control_audits(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(
                _camera_control_audits(child, path=f"{path}[{index}]")
            )
    return result


def _audit_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    nested = value.get("audit")
    payload = nested if isinstance(nested, Mapping) else value
    return deepcopy(dict(payload))


def _safe_token(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in value.lower()
    ).strip("_")
    return result or "metric"


def _kind_counts(items: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("graph records must contain JSON objects")
        kind = str(item.get("kind") or "").strip()
        if not kind:
            raise ValueError("graph records require a non-empty kind")
        result[kind] = result.get(kind, 0) + 1
    return dict(sorted(result.items()))


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
