from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence, runtime_checkable


GROUPING_ROLE = "evidence_partition_not_metric_verdict"
FORBIDDEN_GROUPING_OUTPUT_FIELDS = frozenset(
    {
        "verdict",
        "score",
        "metric_score",
        "applicable",
        "evidence_status",
    }
)


@dataclass(frozen=True)
class GroupingRequest:
    """Implementation-independent request for an object partition.

    Grouping defines the scope of later evidence and metric calls. It does not
    decide whether a scene, group, relation, or arrangement is valid.
    """

    scene: dict[str, Any]
    case: dict[str, Any] = field(default_factory=dict)
    visual_evidence: tuple[Any, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> GroupingRequest:
        if isinstance(value, cls):
            return cls(
                scene=deepcopy(value.scene),
                case=deepcopy(value.case),
                visual_evidence=tuple(deepcopy(value.visual_evidence)),
                config=deepcopy(value.config),
                context=deepcopy(value.context),
            )
        if not isinstance(value, dict):
            raise TypeError("grouping request must be a JSON object")
        scene = value.get("scene")
        if not isinstance(scene, dict):
            raise ValueError("grouping request requires a scene object")
        case = _json_object(value.get("case"), "grouping request case")
        config = _json_object(
            value.get("config"),
            "grouping request config",
        )
        context = _json_object(
            value.get("context"),
            "grouping request context",
        )
        visual_evidence = value.get("visual_evidence")
        if visual_evidence is None:
            visual_evidence = value.get("render_evidence")
        if visual_evidence is None:
            visual_evidence = []
        if not isinstance(visual_evidence, (list, tuple)):
            raise ValueError(
                "grouping request visual_evidence must be a list"
            )
        return cls(
            scene=deepcopy(scene),
            case=case,
            visual_evidence=tuple(deepcopy(visual_evidence)),
            config=config,
            context=context,
        )


@dataclass(frozen=True)
class ObjectGroup:
    group_id: str
    object_ids: tuple[str, ...]
    group_source: str
    label: str | None = None
    anchor_object_id: str | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(self.metadata)
        result.update(
            {
                "group_id": self.group_id,
                "group_source": self.group_source,
                "object_ids": list(self.object_ids),
                "num_objects": len(self.object_ids),
            }
        )
        if self.label is not None:
            result["label"] = self.label
        if self.anchor_object_id is not None:
            result["anchor_object_id"] = self.anchor_object_id
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class GroupingResult:
    """Validated, complete object partition shared by all grouping backends."""

    object_groups: tuple[ObjectGroup, ...]
    backend: str
    policy_id: str
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)
    resolved_grouping_config: dict[str, Any] = field(default_factory=dict)
    omitted_edges: tuple[dict[str, Any], ...] = ()
    cross_group_relations: tuple[dict[str, Any], ...] = ()
    object_catalog: tuple[dict[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        groups: Sequence[dict[str, Any] | ObjectGroup],
        expected_object_ids: Sequence[str],
        backend: str,
        policy_id: str,
        reason: str,
        provenance: dict[str, Any] | None = None,
        resolved_grouping_config: dict[str, Any] | None = None,
        omitted_edges: Sequence[dict[str, Any]] = (),
        cross_group_relations: Sequence[dict[str, Any]] = (),
        object_catalog: Sequence[dict[str, Any]] = (),
    ) -> GroupingResult:
        backend = _required_text(backend, "grouping backend")
        policy_id = _required_text(policy_id, "grouping policy_id")
        reason = _required_text(reason, "grouping reason")
        expected = _unique_text_tuple(
            expected_object_ids,
            "expected grouping object IDs",
        )
        order = {object_id: index for index, object_id in enumerate(expected)}
        normalized: list[ObjectGroup] = []
        for index, value in enumerate(groups):
            group = _object_group_from_value(
                value,
                default_source=backend,
                label=f"object_groups[{index}]",
            )
            normalized.append(group)

        assigned = [
            object_id
            for group in normalized
            for object_id in group.object_ids
        ]
        unknown = sorted(set(assigned) - set(expected))
        if unknown:
            raise ValueError(
                f"grouping result references unknown object IDs {unknown}"
            )
        duplicates = sorted(
            {
                object_id
                for object_id in assigned
                if assigned.count(object_id) > 1
            }
        )
        if duplicates:
            raise ValueError(
                "grouping result assigns object IDs to multiple groups: "
                f"{duplicates}"
            )
        missing = [object_id for object_id in expected if object_id not in assigned]
        if missing:
            raise ValueError(
                "grouping result must assign every renderable object exactly "
                f"once; missing {missing}"
            )
        if not expected and normalized:
            raise ValueError(
                "grouping result cannot contain groups for an empty scene"
            )

        normalized.sort(
            key=lambda group: min(
                order[object_id] for object_id in group.object_ids
            )
        )
        canonical: list[ObjectGroup] = []
        for index, group in enumerate(normalized, start=1):
            members = tuple(
                sorted(group.object_ids, key=order.__getitem__)
            )
            anchor = group.anchor_object_id
            if anchor is not None and anchor not in members:
                raise ValueError(
                    f"group {group.group_id!r} anchor_object_id must be a "
                    "member of the group"
                )
            canonical.append(
                replace(
                    group,
                    group_id=f"group_{index:03d}",
                    object_ids=members,
                )
            )

        return cls(
            object_groups=tuple(canonical),
            backend=backend,
            policy_id=policy_id,
            reason=reason,
            provenance=deepcopy(provenance or {}),
            resolved_grouping_config=deepcopy(
                resolved_grouping_config or {}
            ),
            omitted_edges=tuple(deepcopy(list(omitted_edges))),
            cross_group_relations=tuple(
                deepcopy(list(cross_group_relations))
            ),
            object_catalog=tuple(deepcopy(list(object_catalog))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_groups": [
                group.to_dict() for group in self.object_groups
            ],
            "grouping_role": GROUPING_ROLE,
            "grouping_backend": self.backend,
            "grouping_policy_id": self.policy_id,
            "reason": self.reason,
            "resolved_grouping_config": deepcopy(
                self.resolved_grouping_config
            ),
            "omitted_edges": list(deepcopy(self.omitted_edges)),
            "cross_group_relations": list(
                deepcopy(self.cross_group_relations)
            ),
            "object_catalog": list(deepcopy(self.object_catalog)),
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class GroupingAlgorithm(Protocol):
    backend: str
    policy_id: str

    def group(self, request: GroupingRequest) -> GroupingResult: ...


def reject_metric_outputs(
    value: dict[str, Any],
    *,
    label: str = "grouping response",
) -> None:
    forbidden = sorted(
        key for key in FORBIDDEN_GROUPING_OUTPUT_FIELDS if key in value
    )
    if forbidden:
        raise ValueError(
            f"{label} must not contain metric outputs {forbidden}"
        )


def _object_group_from_value(
    value: dict[str, Any] | ObjectGroup,
    *,
    default_source: str,
    label: str,
) -> ObjectGroup:
    if isinstance(value, ObjectGroup):
        if not value.object_ids:
            raise ValueError(f"{label} cannot be empty")
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    reject_metric_outputs(value, label=label)
    members = value.get("object_ids")
    if not isinstance(members, (list, tuple)) or not members:
        raise ValueError(f"{label}.object_ids must be a non-empty list")
    object_ids = _unique_text_tuple(
        members,
        f"{label}.object_ids",
    )
    group_id = str(value.get("group_id") or f"proposed_{label}").strip()
    source = str(value.get("group_source") or default_source).strip()
    if not source:
        raise ValueError(f"{label}.group_source must be non-empty")
    label_value = value.get("label")
    if label_value is not None:
        label_value = _required_text(label_value, f"{label}.label")
    anchor = value.get("anchor_object_id")
    if anchor is not None:
        anchor = _required_text(anchor, f"{label}.anchor_object_id")
    reason = str(value.get("reason") or "").strip()
    reserved = {
        "group_id",
        "group_source",
        "object_ids",
        "num_objects",
        "label",
        "anchor_object_id",
        "reason",
    }
    metadata = {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in reserved
    }
    return ObjectGroup(
        group_id=group_id,
        object_ids=object_ids,
        group_source=source,
        label=label_value,
        anchor_object_id=anchor,
        reason=reason,
        metadata=metadata,
    )


def _required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _unique_text_tuple(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    result: list[str] = []
    for value in values:
        if not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"{label} must contain non-empty object IDs")
        result.append(str(value).strip())
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate object IDs")
    return tuple(result)


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return deepcopy(value)
