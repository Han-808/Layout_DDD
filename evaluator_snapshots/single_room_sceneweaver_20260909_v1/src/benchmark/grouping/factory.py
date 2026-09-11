from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from benchmark.grouping.anchor import AnchorGroupingAlgorithm
from benchmark.grouping.fallback import (
    VLMPrimaryGroupingAlgorithm,
    resolve_grouping_fallback_config,
)
from benchmark.grouping.interfaces import (
    GroupingAlgorithm,
    GroupingRequest,
    GroupingResult,
)
from benchmark.grouping.topology import TopologyGroupingAlgorithm
from benchmark.grouping.vlm import VLMGroupingAlgorithm


# All three names remain accepted by the compatibility factory. VLM remains the
# active/default primary. Topology is also the sole validated runtime recovery
# backend; anchor remains available only for explicit historical replay.
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
    fallback_config = resolve_grouping_fallback_config(resolved)
    if model is None and not fallback_config["enabled"]:
        raise ValueError(
            "grouping backend 'vlm' requires an injected chat model"
        )
    primary_error: BaseException | None = None
    primary: GroupingAlgorithm | None = None
    if model is None:
        primary_error = ValueError(
            "grouping backend 'vlm' requires an injected chat model"
        )
    else:
        primary = VLMGroupingAlgorithm(model, resolved)
    fallback = TopologyGroupingAlgorithm(
        _topology_fallback_config(resolved)
    )
    return VLMPrimaryGroupingAlgorithm(
        primary=primary,
        fallback=fallback,
        primary_policy_id=VLMGroupingAlgorithm.policy_id,
        fallback_config=fallback_config,
        primary_unavailable_error=primary_error,
    )


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


def _topology_fallback_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    section = config.get("grouping", config)
    if not isinstance(section, dict):
        return {"grouping": {"backend": "topology"}}
    topology = section.get("topology")
    result: dict[str, Any] = {
        "grouping": {"backend": "topology"}
    }
    if isinstance(topology, dict):
        result["grouping"]["topology"] = deepcopy(topology)
    return result
