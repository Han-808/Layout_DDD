from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Protocol, runtime_checkable

from benchmark.visual_judge.camera_dsl import (
    CameraConstraintSet,
    camera_constraints_from_judge_request,
)
from benchmark.visual_judge.interfaces.judge import EvidenceRequest


@dataclass(frozen=True)
class MetricAcquisitionPlanningRequest:
    metric: str
    evidence_request: EvidenceRequest
    known_target_ids: tuple[str, ...]
    relation_type: str | None = None


@runtime_checkable
class EvidenceAcquisitionPlanner(Protocol):
    def plan(
        self,
        request: MetricAcquisitionPlanningRequest,
    ) -> CameraConstraintSet: ...


class MetricSpecificAcquisitionPlanner:
    """Translate Judge observations into validated Camera DSL constraints."""

    backend = "metric_specific_camera_dsl"

    def plan(
        self,
        request: MetricAcquisitionPlanningRequest,
    ) -> CameraConstraintSet:
        if not isinstance(request, MetricAcquisitionPlanningRequest):
            raise TypeError(
                "acquisition planner requires "
                "MetricAcquisitionPlanningRequest"
            )
        constraints = camera_constraints_from_judge_request(
            request.evidence_request,
            metric=request.metric,
            known_target_ids=_target_tuple(request.known_target_ids),
            relation_type=request.relation_type,
        )
        return replace(
            constraints,
            metadata={
                **constraints.metadata,
                "planner_backend": self.backend,
            },
        )


def _target_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value) for value in values if str(value).strip())
    )
