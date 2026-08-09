"""Validated data model for read-only evaluation query graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from benchmark.visual_judge.graphs._validation import (
    json_object,
    required_text,
)


EVALUATION_QUERY_GRAPH_VERSION = "evaluation_query_graph_v1"
QUERY_NODE_KINDS = frozenset(
    {
        "evaluation",
        "metric",
        "claim",
        "scope",
        "object",
        "relation_candidate",
        "evidence_artifact",
        "evidence_request",
        "acquisition_episode",
        "camera_selection",
        "judge_call",
        "typed_check",
        "check_result",
        "ownership_event",
        "decision",
    }
)
QUERY_EDGE_KINDS = frozenset(
    {
        "contains_metric",
        "contains_claim",
        "under_metric",
        "targets",
        "scoped_to",
        "uses_evidence",
        "judged_by",
        "requests_evidence",
        "starts_episode",
        "contains_episode",
        "contains_selection",
        "produces_evidence",
        "produces_decision",
        "examines_relation",
        "relates",
        "relation_has_evidence",
        "routes_to_check",
        "requires_check",
        "defers_check",
        "check_targets",
        "check_context",
        "check_uses_evidence",
        "produces_check_result",
        "resolves_check",
        "excluded_by_ownership",
        "ownership_affects",
        "ownership_caused_by",
        "ownership_scored_to",
        "supports_decision",
    }
)
EDGE_ENDPOINT_KINDS = {
    "contains_metric": ({"evaluation"}, {"metric"}),
    "contains_claim": ({"evaluation"}, {"claim"}),
    "under_metric": ({"claim"}, {"metric"}),
    "targets": ({"claim", "evidence_request"}, {"object"}),
    "scoped_to": (
        {
            "claim",
            "evidence_request",
            "relation_candidate",
            "typed_check",
        },
        {"scope"},
    ),
    "uses_evidence": ({"claim", "judge_call"}, {"evidence_artifact"}),
    "judged_by": ({"claim"}, {"judge_call"}),
    "requests_evidence": ({"judge_call"}, {"evidence_request"}),
    "starts_episode": ({"evidence_request"}, {"acquisition_episode"}),
    "contains_episode": ({"evaluation"}, {"acquisition_episode"}),
    "contains_selection": (
        {"acquisition_episode"},
        {"camera_selection"},
    ),
    "produces_evidence": (
        {"acquisition_episode"},
        {"evidence_artifact"},
    ),
    "produces_decision": ({"judge_call"}, {"decision"}),
    "examines_relation": ({"claim"}, {"relation_candidate"}),
    "relates": ({"relation_candidate"}, {"object"}),
    "relation_has_evidence": (
        {"relation_candidate"},
        {"evidence_artifact"},
    ),
    "routes_to_check": ({"relation_candidate"}, {"typed_check"}),
    "requires_check": ({"claim"}, {"typed_check"}),
    "defers_check": ({"claim"}, {"typed_check"}),
    "check_targets": ({"typed_check"}, {"object"}),
    "check_context": ({"typed_check"}, {"object"}),
    "check_uses_evidence": (
        {"typed_check"},
        {"evidence_artifact"},
    ),
    "produces_check_result": ({"judge_call"}, {"check_result"}),
    "resolves_check": ({"check_result"}, {"typed_check"}),
    "excluded_by_ownership": (
        {"check_result"},
        {"ownership_event"},
    ),
    "ownership_affects": ({"ownership_event"}, {"object"}),
    "ownership_caused_by": ({"ownership_event"}, {"object"}),
    "ownership_scored_to": ({"ownership_event"}, {"object"}),
    "supports_decision": ({"check_result"}, {"decision"}),
}


@dataclass(frozen=True)
class QueryGraphNode:
    node_id: str
    kind: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text(self.node_id, "query node_id")
        required_text(self.label, "query node label")
        if self.kind not in QUERY_NODE_KINDS:
            raise ValueError(f"unsupported query node kind {self.kind!r}")
        json_object(self.attributes, "query node attributes")
        json_object(self.provenance, "query node provenance")

    @classmethod
    def from_value(cls, value: Any) -> QueryGraphNode:
        if isinstance(value, cls):
            return value
        _require_mapping(value, "query graph node")
        _reject_unknown(
            value,
            {"node_id", "kind", "label", "attributes", "provenance"},
            "query graph node",
        )
        return cls(
            node_id=str(value.get("node_id") or "").strip(),
            kind=str(value.get("kind") or "").strip(),
            label=str(value.get("label") or "").strip(),
            attributes=json_object(
                value.get("attributes"), "query node attributes"
            ),
            provenance=json_object(
                value.get("provenance"), "query node provenance"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "attributes": deepcopy(self.attributes),
            "provenance": deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class QueryGraphEdge:
    edge_id: str
    kind: str
    source_id: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text(self.edge_id, "query edge_id")
        source = required_text(self.source_id, "query edge source_id")
        target = required_text(self.target_id, "query edge target_id")
        if self.kind not in QUERY_EDGE_KINDS:
            raise ValueError(f"unsupported query edge kind {self.kind!r}")
        if source == target:
            raise ValueError("query graph edges cannot be self-referential")
        json_object(self.attributes, "query edge attributes")
        json_object(self.provenance, "query edge provenance")

    @classmethod
    def from_value(cls, value: Any) -> QueryGraphEdge:
        if isinstance(value, cls):
            return value
        _require_mapping(value, "query graph edge")
        _reject_unknown(
            value,
            {
                "edge_id",
                "kind",
                "source_id",
                "target_id",
                "attributes",
                "provenance",
            },
            "query graph edge",
        )
        return cls(
            edge_id=str(value.get("edge_id") or "").strip(),
            kind=str(value.get("kind") or "").strip(),
            source_id=str(value.get("source_id") or "").strip(),
            target_id=str(value.get("target_id") or "").strip(),
            attributes=json_object(
                value.get("attributes"), "query edge attributes"
            ),
            provenance=json_object(
                value.get("provenance"), "query edge provenance"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "attributes": deepcopy(self.attributes),
            "provenance": deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class EvaluationQueryGraph:
    graph_id: str
    case_id: str
    metric: str
    nodes: tuple[QueryGraphNode, ...]
    edges: tuple[QueryGraphEdge, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EVALUATION_QUERY_GRAPH_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_QUERY_GRAPH_VERSION:
            raise ValueError("unsupported evaluation query graph version")
        required_text(self.graph_id, "evaluation query graph_id")
        required_text(self.case_id, "evaluation query case_id")
        required_text(self.metric, "evaluation query metric")
        nodes: dict[str, QueryGraphNode] = {}
        for node in self.nodes:
            if not isinstance(node, QueryGraphNode):
                raise TypeError("query graph contains an invalid node")
            if node.node_id in nodes:
                raise ValueError("query graph node IDs must be unique")
            nodes[node.node_id] = node
        for kind in ("evaluation", "metric", "claim"):
            if not any(node.kind == kind for node in self.nodes):
                raise ValueError(f"query graph requires a {kind} node")
        edge_ids: set[str] = set()
        for edge in self.edges:
            if not isinstance(edge, QueryGraphEdge):
                raise TypeError("query graph contains an invalid edge")
            if edge.edge_id in edge_ids:
                raise ValueError("query graph edge IDs must be unique")
            edge_ids.add(edge.edge_id)
            if edge.source_id not in nodes or edge.target_id not in nodes:
                raise ValueError("query graph edge references unknown node")
            sources, targets = EDGE_ENDPOINT_KINDS[edge.kind]
            source_kind = nodes[edge.source_id].kind
            target_kind = nodes[edge.target_id].kind
            if source_kind not in sources or target_kind not in targets:
                raise ValueError(
                    f"query edge {edge.kind!r} cannot connect "
                    f"{source_kind!r} to {target_kind!r}"
                )
        json_object(self.metadata, "evaluation query metadata")

    @classmethod
    def from_value(cls, value: Any) -> EvaluationQueryGraph:
        if isinstance(value, cls):
            return value
        _require_mapping(value, "evaluation query graph")
        _reject_unknown(
            value,
            {
                "schema_version",
                "graph_id",
                "case_id",
                "metric",
                "nodes",
                "edges",
                "metadata",
                "decision_authority",
                "projection_mode",
            },
            "evaluation query graph",
        )
        if value.get("decision_authority") not in (None, "none"):
            raise ValueError("evaluation query graph has no decision authority")
        if value.get("projection_mode") not in (None, "posthoc_read_only"):
            raise ValueError("evaluation query graph must be posthoc_read_only")
        raw_nodes = value.get("nodes")
        raw_edges = value.get("edges")
        if not isinstance(raw_nodes, (list, tuple)):
            raise ValueError("query graph nodes must be a JSON list")
        if not isinstance(raw_edges, (list, tuple)):
            raise ValueError("query graph edges must be a JSON list")
        return cls(
            schema_version=str(
                value.get("schema_version") or EVALUATION_QUERY_GRAPH_VERSION
            ),
            graph_id=str(value.get("graph_id") or "").strip(),
            case_id=str(value.get("case_id") or "").strip(),
            metric=str(value.get("metric") or "").strip(),
            nodes=tuple(QueryGraphNode.from_value(item) for item in raw_nodes),
            edges=tuple(QueryGraphEdge.from_value(item) for item in raw_edges),
            metadata=json_object(
                value.get("metadata"), "evaluation query metadata"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "case_id": self.case_id,
            "metric": self.metric,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": deepcopy(self.metadata),
            "decision_authority": "none",
            "projection_mode": "posthoc_read_only",
        }


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
