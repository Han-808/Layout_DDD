from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from benchmark.grouping.anchor import AnchorGroupingAlgorithm
from benchmark.grouping.interfaces import (
    GroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
)
from benchmark.grouping.topology import TopologyGroupingAlgorithm
from benchmark.grouping.vlm import VLMGroupingAlgorithm


# All three names remain accepted by the compatibility factory. Only VLM is an
# active/default grouping implementation; topology and anchor are retained for
# explicit historical replay and are never selected as a fallback.
GROUPING_BACKENDS = ("topology", "anchor", "vlm")
ACTIVE_GROUPING_BACKENDS = ("vlm",)
DEPRECATED_GROUPING_BACKENDS = ("topology", "anchor")
DEFAULT_GROUPING_BACKEND = "vlm"


def build_grouping_algorithm(
    config: dict[str, Any] | None = None,
    *,
    model: Any | None = None,
) -> GroupingAlgorithm:
    if config is not None and not isinstance(config, dict):
        raise TypeError("grouping config must be a JSON object")
    resolved = deepcopy(config or {})
    section = resolved.get("grouping")
    if isinstance(section, dict):
        section = deepcopy(section)
    else:
        section = resolved
    backend = str(
        section.get("backend") or DEFAULT_GROUPING_BACKEND
    ).strip()
    if backend not in GROUPING_BACKENDS:
        raise ValueError(
            f"grouping backend must be one of {list(GROUPING_BACKENDS)}"
        )
    if backend == "topology":
        return TopologyGroupingAlgorithm(resolved)
    if backend == "anchor":
        return AnchorGroupingAlgorithm(resolved)
    if model is None:
        raise ValueError(
            "grouping backend 'vlm' requires an injected chat model"
        )
    return VLMGroupingAlgorithm(model, resolved)


def group_scene(
    scene: dict[str, Any],
    *,
    case: dict[str, Any] | None = None,
    visual_evidence: Iterable[Any] = (),
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    model: Any | None = None,
    algorithm: GroupingAlgorithm | None = None,
) -> GroupingResult:
    """Partition one scene with an explicit, replaceable grouping backend."""

    selected = algorithm or build_grouping_algorithm(
        config,
        model=model,
    )
    if not callable(getattr(selected, "group", None)):
        raise TypeError(
            "group_scene requires an algorithm with group(request)"
        )
    result = selected.group(
        GroupingRequest(
            scene=deepcopy(scene),
            case=deepcopy(case or {}),
            visual_evidence=tuple(deepcopy(list(visual_evidence))),
            config=deepcopy(config or {}),
            context=deepcopy(context or {}),
        )
    )
    if not isinstance(result, GroupingResult):
        raise TypeError(
            "grouping algorithm must return a GroupingResult"
        )
    return result
