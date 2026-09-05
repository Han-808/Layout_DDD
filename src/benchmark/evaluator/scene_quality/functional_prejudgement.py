"""Pluggable pre-Judge functional evidence preparation.

This stage is deliberately separate from the Controller's Judge-triggered
camera acquisition.  The runtime implementation wraps the existing functional
probe pipeline without changing its ordering or budget semantics; disabled and
frozen implementations support controlled ablations.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from benchmark.evaluator.scene_quality.functional_acquisition import (
    FUNCTIONAL_ACQUISITION_PLAN_VERSION,
)
from benchmark.evaluator.scene_quality.functional_planner_adapter import (
    is_functional_discovery_planner_mode,
)
from benchmark.evaluator.scene_quality.functional_probe import (
    FUNCTIONAL_PROBE_ACQUISITION_VERSION,
    acquire_functional_probe_evidence,
    functional_probe_judge_packet,
)
from benchmark.visual_judge.identity_evidence import (
    validate_identity_evidence,
)
from benchmark.visual_judge.usable_surface import (
    DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND,
)

_COMPATIBLE_FUNCTIONAL_PROBE_ACQUISITION_VERSIONS = frozenset(
    {
        "functional_probe_acquisition_v3",
        "functional_probe_acquisition_v4",
        FUNCTIONAL_PROBE_ACQUISITION_VERSION,
    }
)
_COMPATIBLE_FUNCTIONAL_ACQUISITION_PLAN_VERSIONS = frozenset(
    {
        "functional_acquisition_plan_v7",
        "functional_acquisition_plan_v8",
        "functional_acquisition_plan_v9",
        FUNCTIONAL_ACQUISITION_PLAN_VERSION,
    }
)


FUNCTIONAL_PREJUDGEMENT_EVIDENCE_VERSION = (
    "functional_prejudgement_evidence_v1"
)
FUNCTIONAL_PREJUDGEMENT_EVIDENCE_MODES = frozenset(
    {"runtime", "disabled", "frozen"}
)
DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG: dict[str, Any] = {
    "mode": "runtime",
    "frozen_result": None,
    "frozen_result_path": None,
    "expected_usable_surface_detector": None,
}


@dataclass(frozen=True)
class FunctionalPrejudgementEvidenceRequest:
    """Immutable identity and inputs for proactive Functional evidence."""

    scene: dict[str, Any]
    global_image_path: str
    max_probe_units: int
    groups: tuple[dict[str, Any], ...] = ()
    grouping_report: dict[str, Any] | None = None
    scene_id: str | None = None
    scene_sha256: str = ""
    global_image_sha256: str = ""
    object_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    identity_image_path: str | None = None
    identity_image_sha256: str | None = None
    identity_legend: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        scene: dict[str, Any],
        global_image_path: str,
        max_probe_units: int,
        groups: list[dict[str, Any]] | None,
        grouping_report: dict[str, Any] | None,
        identity_image_path: str | None = None,
        identity_legend: dict[str, str] | None = None,
    ) -> "FunctionalPrejudgementEvidenceRequest":
        if not isinstance(scene, dict):
            raise TypeError(
                "functional prejudgement scene must be a JSON object"
            )
        image_path = Path(global_image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                "functional prejudgement global image does not exist: "
                f"{image_path}"
            )
        if (
            isinstance(max_probe_units, bool)
            or not isinstance(max_probe_units, int)
            or max_probe_units < 0
        ):
            raise ValueError(
                "functional prejudgement max_probe_units must be "
                "a non-negative integer"
            )
        normalized_groups = tuple(
            deepcopy(item)
            for item in groups or []
            if isinstance(item, dict)
        )
        object_ids = tuple(
            str(item.get("id"))
            for item in scene.get("objects") or []
            if isinstance(item, dict)
            and str(item.get("id") or "").strip()
        )
        group_ids = tuple(
            str(item.get("group_id"))
            for item in normalized_groups
            if str(item.get("group_id") or "").strip()
        )
        identity = validate_identity_evidence(
            image_path=identity_image_path,
            legend=identity_legend,
            expected_object_ids=object_ids,
            label="functional prejudgement",
        )
        normalized_identity_path = identity["identity_image_path"]
        return cls(
            scene=deepcopy(scene),
            global_image_path=str(image_path),
            max_probe_units=max_probe_units,
            groups=normalized_groups,
            grouping_report=deepcopy(grouping_report),
            scene_id=(
                str(scene.get("scene_id"))
                if scene.get("scene_id") is not None
                else None
            ),
            scene_sha256=_canonical_json_sha256(scene),
            global_image_sha256=_content_sha256(image_path),
            object_ids=object_ids,
            group_ids=group_ids,
            identity_image_path=normalized_identity_path,
            identity_image_sha256=(
                _content_sha256(Path(normalized_identity_path))
                if normalized_identity_path is not None
                else None
            ),
            identity_legend=deepcopy(identity["identity_legend"]),
        )

    def identity(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_sha256": self.scene_sha256,
            "global_image_path": self.global_image_path,
            "global_image_sha256": self.global_image_sha256,
            "object_ids": list(self.object_ids),
            "group_ids": list(self.group_ids),
            "identity_image_path": self.identity_image_path,
            "identity_image_sha256": self.identity_image_sha256,
            "identity_legend": deepcopy(self.identity_legend),
        }


@dataclass(frozen=True)
class FunctionalPrejudgementEvidenceResult:
    """Validated, decision-free output consumed by the Functional Judge."""

    status: str
    selected_judge_probe_paths: tuple[str, ...] = ()
    cross_group_probe_paths: tuple[str, ...] = ()
    cross_group_probe_packet: dict[str, Any] | None = None
    group_owned_probe_packets: dict[str, Any] = field(
        default_factory=dict
    )
    functional_discovery: dict[str, Any] | None = None
    usable_surface_hypotheses: tuple[dict[str, Any], ...] = ()
    functional_boundary_evidence: dict[str, Any] | None = None
    acquisition_plan: dict[str, Any] | None = None
    unscheduled_discovery_items: tuple[dict[str, Any], ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)
    budget_usage: dict[str, Any] = field(default_factory=dict)
    source_identity: dict[str, Any] = field(default_factory=dict)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    runtime_audit: dict[str, Any] = field(default_factory=dict)
    decision_authority: str = "none"
    schema_version: str = FUNCTIONAL_PREJUDGEMENT_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.decision_authority != "none":
            raise ValueError(
                "functional prejudgement evidence has no decision authority"
            )
        if self.schema_version != FUNCTIONAL_PREJUDGEMENT_EVIDENCE_VERSION:
            raise ValueError(
                "unsupported functional prejudgement evidence schema"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "selected_judge_probe_paths": list(
                self.selected_judge_probe_paths
            ),
            "cross_group_probe_paths": list(
                self.cross_group_probe_paths
            ),
            "cross_group_probe_packet": deepcopy(
                self.cross_group_probe_packet
            ),
            "group_owned_probe_packets": deepcopy(
                self.group_owned_probe_packets
            ),
            "functional_discovery": deepcopy(
                self.functional_discovery
            ),
            "usable_surface_hypotheses": list(
                deepcopy(self.usable_surface_hypotheses)
            ),
            "functional_boundary_evidence": deepcopy(
                self.functional_boundary_evidence
            ),
            "acquisition_plan": deepcopy(self.acquisition_plan),
            "unscheduled_discovery_items": list(
                deepcopy(self.unscheduled_discovery_items)
            ),
            "telemetry": deepcopy(self.telemetry),
            "budget_usage": deepcopy(self.budget_usage),
            "source_identity": deepcopy(self.source_identity),
            "artifact_sha256": deepcopy(self.artifact_sha256),
            "provenance": deepcopy(self.provenance),
            "runtime_audit": deepcopy(self.runtime_audit),
            "decision_authority": self.decision_authority,
        }


@runtime_checkable
class FunctionalPrejudgementEvidenceSource(Protocol):
    """A decision-free proactive Functional evidence source."""

    mode: str

    def prepare_functional_evidence(
        self,
        request: FunctionalPrejudgementEvidenceRequest,
    ) -> FunctionalPrejudgementEvidenceResult:
        ...

    def manifest(self) -> dict[str, Any]:
        ...


class RuntimeFunctionalPrejudgementEvidenceSource:
    """Compatibility wrapper around the current end-to-end probe pipeline."""

    mode = "runtime"

    def __init__(self, *, planner: Any, provider: Any) -> None:
        self.planner = planner
        self.provider = provider

    def prepare_functional_evidence(
        self,
        request: FunctionalPrejudgementEvidenceRequest,
    ) -> FunctionalPrejudgementEvidenceResult:
        paths, audit = acquire_functional_probe_evidence(
            planner=self.planner,
            provider=self.provider,
            scene=deepcopy(request.scene),
            global_image_path=request.global_image_path,
            max_probe_units=request.max_probe_units,
            groups=list(deepcopy(request.groups)),
            grouping_report=deepcopy(request.grouping_report),
            identity_image_path=request.identity_image_path,
            identity_legend=deepcopy(request.identity_legend),
        )
        return _runtime_result(
            request=request,
            paths=paths,
            audit=audit,
            source_manifest=self.manifest(),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "planner": _component_identity(self.planner),
            "provider": _component_identity(self.provider),
            "usable_surface_detector": _provider_detector_manifest(
                self.provider
            ),
            "decision_authority": "none",
        }


class DisabledFunctionalPrejudgementEvidenceSource:
    """No-op source for isolating Judge-triggered camera acquisition."""

    mode = "disabled"

    def prepare_functional_evidence(
        self,
        request: FunctionalPrejudgementEvidenceRequest,
    ) -> FunctionalPrejudgementEvidenceResult:
        audit = {
            "schema_version": FUNCTIONAL_PROBE_ACQUISITION_VERSION,
            "status": "disabled",
            "reason": "functional_prejudgement_evidence_disabled",
            "decision_authority": "none",
            "selected_raw_rgb_paths": [],
            "cross_group_evidence_paths": [],
            "group_evidence_paths": {},
            "group_probe_packets": {},
            "unscheduled_discovery_items": [],
            "probe_results": [],
            "coverage_complete": True,
            "budget_exhausted": False,
            "source_scene_modified": False,
        }
        return FunctionalPrejudgementEvidenceResult(
            status="disabled",
            source_identity=request.identity(),
            telemetry=_empty_stage_telemetry(),
            budget_usage={
                "max_probe_units": request.max_probe_units,
                "scheduled_probe_count": 0,
                "unscheduled_discovery_count": 0,
                "budget_exhausted": False,
            },
            provenance=self.manifest(),
            runtime_audit=audit,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "decision_authority": "none",
        }


class FrozenFunctionalPrejudgementEvidenceSource:
    """Strictly reuse one previously generated proactive evidence result."""

    mode = "frozen"

    def __init__(
        self,
        result: dict[str, Any],
        *,
        expected_detector_implementation_id: str,
        expected_detector_version: str,
    ) -> None:
        if not isinstance(result, dict):
            raise TypeError(
                "frozen functional prejudgement result must be an object"
            )
        self._result = deepcopy(result)
        self.expected_detector_implementation_id = str(
            expected_detector_implementation_id
        ).strip()
        self.expected_detector_version = str(
            expected_detector_version
        ).strip()
        if (
            not self.expected_detector_implementation_id
            or not self.expected_detector_version
        ):
            raise ValueError(
                "frozen functional prejudgement evidence requires expected "
                "usable-surface detector identity and version"
            )

    def prepare_functional_evidence(
        self,
        request: FunctionalPrejudgementEvidenceRequest,
    ) -> FunctionalPrejudgementEvidenceResult:
        return validate_frozen_functional_prejudgement_result(
            self._result,
            request=request,
            expected_detector_implementation_id=(
                self.expected_detector_implementation_id
            ),
            expected_detector_version=self.expected_detector_version,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "expected_usable_surface_detector": {
                "implementation_id": (
                    self.expected_detector_implementation_id
                ),
                "version": self.expected_detector_version,
            },
            "decision_authority": "none",
        }


def resolve_functional_prejudgement_evidence_source(
    config: dict[str, Any] | None,
    *,
    planner: Any,
    provider: Any,
    injected_source: Any = None,
) -> FunctionalPrejudgementEvidenceSource:
    """Resolve one source without falling back between implementations."""

    resolved = validate_functional_prejudgement_evidence_config(
        config
    )
    if injected_source is not None:
        if (
            not callable(
                getattr(
                    injected_source,
                    "prepare_functional_evidence",
                    None,
                )
            )
            or not callable(getattr(injected_source, "manifest", None))
            or str(getattr(injected_source, "mode", "")).strip()
            not in FUNCTIONAL_PREJUDGEMENT_EVIDENCE_MODES
        ):
            raise TypeError(
                "injected functional prejudgement source must expose a "
                "supported mode, prepare_functional_evidence(request), "
                "and manifest()"
            )
        manifest = injected_source.manifest()
        if (
            not isinstance(manifest, dict)
            or manifest.get("decision_authority") != "none"
        ):
            raise ValueError(
                "functional prejudgement source manifest must declare "
                "decision_authority=none"
            )
        return injected_source
    mode = resolved["mode"]
    if mode == "runtime":
        return RuntimeFunctionalPrejudgementEvidenceSource(
            planner=planner,
            provider=provider,
        )
    if mode == "disabled":
        return DisabledFunctionalPrejudgementEvidenceSource()
    frozen = resolved.get("frozen_result")
    frozen_path = resolved.get("frozen_result_path")
    if frozen is None and frozen_path is not None:
        path = Path(str(frozen_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "frozen functional prejudgement result does not exist: "
                f"{path}"
            )
        frozen = json.loads(path.read_text(encoding="utf-8"))
    detector = resolved["expected_usable_surface_detector"]
    return FrozenFunctionalPrejudgementEvidenceSource(
        frozen,
        expected_detector_implementation_id=detector[
            "implementation_id"
        ],
        expected_detector_version=detector["version"],
    )


def validate_functional_prejudgement_evidence_config(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return deepcopy(
            DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG
        )
    if not isinstance(value, dict):
        raise TypeError(
            "functional_prejudgement_evidence must be a JSON object"
        )
    unknown = set(value) - set(
        DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG
    )
    if unknown:
        raise ValueError(
            "functional_prejudgement_evidence contains unknown fields: "
            f"{sorted(unknown)}"
        )
    resolved = {
        **deepcopy(DEFAULT_FUNCTIONAL_PREJUDGEMENT_EVIDENCE_CONFIG),
        **deepcopy(value),
    }
    mode = str(resolved.get("mode") or "").strip()
    if mode == "precomputed":
        mode = "frozen"
    if mode not in FUNCTIONAL_PREJUDGEMENT_EVIDENCE_MODES:
        raise ValueError(
            "functional_prejudgement_evidence.mode must be runtime, "
            "disabled, or frozen"
        )
    resolved["mode"] = mode
    if mode != "frozen":
        if (
            resolved.get("frozen_result") is not None
            or resolved.get("frozen_result_path") is not None
            or resolved.get("expected_usable_surface_detector")
            is not None
        ):
            raise ValueError(
                "frozen functional prejudgement fields require mode=frozen"
            )
        return resolved
    has_result = resolved.get("frozen_result") is not None
    has_path = resolved.get("frozen_result_path") is not None
    if has_result == has_path:
        raise ValueError(
            "mode=frozen requires exactly one of frozen_result or "
            "frozen_result_path"
        )
    detector = resolved.get("expected_usable_surface_detector")
    if (
        not isinstance(detector, dict)
        or set(detector) != {"implementation_id", "version"}
        or not str(detector.get("implementation_id") or "").strip()
        or not str(detector.get("version") or "").strip()
    ):
        raise ValueError(
            "mode=frozen requires expected_usable_surface_detector with "
            "implementation_id and version"
        )
    return resolved


def validate_frozen_functional_prejudgement_result(
    value: dict[str, Any],
    *,
    request: FunctionalPrejudgementEvidenceRequest,
    expected_detector_implementation_id: str,
    expected_detector_version: str,
) -> FunctionalPrejudgementEvidenceResult:
    """Fail closed on any frozen identity, version, or artifact mismatch."""

    if value.get("schema_version") != (
        FUNCTIONAL_PREJUDGEMENT_EVIDENCE_VERSION
    ):
        raise ValueError(
            "frozen functional prejudgement schema version mismatch"
        )
    if value.get("decision_authority") != "none":
        raise ValueError(
            "frozen functional prejudgement evidence must have "
            "decision_authority=none"
        )
    source_identity = value.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError(
            "frozen functional prejudgement source_identity is required"
        )
    expected_identity = request.identity()
    identity_keys = [
        "scene_id",
        "scene_sha256",
        "global_image_sha256",
        "object_ids",
        "group_ids",
    ]
    if request.identity_image_sha256 is not None:
        identity_keys.extend(
            ("identity_image_sha256", "identity_legend")
        )
    for key in identity_keys:
        if source_identity.get(key) != expected_identity.get(key):
            raise ValueError(
                "frozen functional prejudgement source mismatch: "
                f"{key}"
            )
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "frozen functional prejudgement provenance is required"
        )
    if provenance.get("acquisition_plan_version") not in (
        _COMPATIBLE_FUNCTIONAL_ACQUISITION_PLAN_VERSIONS
    ):
        raise ValueError(
            "frozen functional prejudgement planner version mismatch"
        )
    detector = provenance.get("usable_surface_detector")
    if not isinstance(detector, dict):
        raise ValueError(
            "frozen functional prejudgement detector provenance is required"
        )
    if str(detector.get("implementation_id") or "") != str(
        expected_detector_implementation_id
    ):
        raise ValueError(
            "frozen functional prejudgement detector identity mismatch"
        )
    if str(detector.get("version") or "") != str(
        expected_detector_version
    ):
        raise ValueError(
            "frozen functional prejudgement detector version mismatch"
        )
    paths = _validated_path_list(
        value.get("selected_judge_probe_paths"),
        label="selected_judge_probe_paths",
    )
    cross_group_paths = _validated_path_list(
        value.get("cross_group_probe_paths"),
        label="cross_group_probe_paths",
    )
    cross_group_packet = value.get("cross_group_probe_packet")
    if (
        not isinstance(cross_group_packet, dict)
        or cross_group_packet.get("planning_role")
        != "visual_evidence_only_no_metric_verdict"
    ):
        raise ValueError(
            "frozen functional prejudgement cross-group packet must be "
            "decision-free visual evidence"
        )
    group_packets = value.get("group_owned_probe_packets")
    if not isinstance(group_packets, dict):
        raise ValueError(
            "frozen functional prejudgement group packets must be an object"
        )
    unknown_groups = sorted(
        set(str(key) for key in group_packets)
        - set(request.group_ids)
    )
    if unknown_groups:
        raise ValueError(
            "frozen functional prejudgement contains unknown group IDs: "
            f"{unknown_groups}"
        )
    if any(
        not isinstance(packet, dict)
        for packet in group_packets.values()
    ):
        raise ValueError(
            "frozen functional prejudgement group packets must be objects"
        )
    if any(
        str(packet.get("group_id") or group_id) != str(group_id)
        for group_id, packet in group_packets.items()
    ):
        raise ValueError(
            "frozen functional prejudgement group packet identity mismatch"
        )
    if any(
        packet.get("planning_role")
        != "visual_evidence_only_no_metric_verdict"
        for packet in group_packets.values()
    ):
        raise ValueError(
            "frozen functional prejudgement group packets must be "
            "decision-free visual evidence"
        )
    runtime_audit = value.get("runtime_audit")
    if not isinstance(runtime_audit, dict):
        raise ValueError(
            "frozen functional prejudgement runtime_audit is required"
        )
    if runtime_audit.get("decision_authority") != "none":
        raise ValueError(
            "frozen functional prejudgement runtime audit must have "
            "decision_authority=none"
        )
    audit_schema_version = str(
        runtime_audit.get("schema_version") or ""
    )
    if audit_schema_version not in (
        _COMPATIBLE_FUNCTIONAL_PROBE_ACQUISITION_VERSIONS
    ):
        raise ValueError(
            "frozen functional prejudgement acquisition schema version "
            "mismatch"
        )
    audit_selected_paths = _validated_path_list(
        runtime_audit.get("selected_raw_rgb_paths"),
        label="runtime_audit.selected_raw_rgb_paths",
    )
    audit_cross_group_paths = _validated_path_list(
        runtime_audit.get("cross_group_evidence_paths"),
        label="runtime_audit.cross_group_evidence_paths",
    )
    if audit_selected_paths != paths:
        raise ValueError(
            "frozen functional prejudgement selected path audit mismatch"
        )
    if audit_cross_group_paths != cross_group_paths:
        raise ValueError(
            "frozen functional prejudgement cross-group path audit mismatch"
        )
    if runtime_audit.get("group_probe_packets") != group_packets:
        raise ValueError(
            "frozen functional prejudgement group packet audit mismatch"
        )
    packet_paths_by_group = {
        str(group_id): list(
            dict.fromkeys(
                str(item.get("artifact_id"))
                for item in packet.get("image_order") or []
                if isinstance(item, dict)
                and str(item.get("artifact_id") or "").strip()
            )
        )
        for group_id, packet in group_packets.items()
    }
    audited_group_paths = runtime_audit.get("group_evidence_paths")
    if not isinstance(audited_group_paths, dict):
        raise ValueError(
            "frozen functional prejudgement group path audit is required"
        )
    normalized_audited_group_paths: dict[str, list[str]] = {}
    for group_id, raw_paths in audited_group_paths.items():
        normalized_audited_group_paths[str(group_id)] = (
            _validated_path_list(
                raw_paths,
                label=(
                    "runtime_audit.group_evidence_paths."
                    f"{group_id}"
                ),
            )
        )
    if {
        key: value
        for key, value in normalized_audited_group_paths.items()
        if value
    } != {
        key: value
        for key, value in packet_paths_by_group.items()
        if value
    }:
        raise ValueError(
            "frozen functional prejudgement group evidence path mismatch"
        )
    packet_paths = [
        path
        for paths_for_group in packet_paths_by_group.values()
        for path in paths_for_group
    ]
    selected_path_set = set(paths)
    if not set(cross_group_paths).issubset(selected_path_set):
        raise ValueError(
            "frozen functional prejudgement cross-group paths must be "
            "selected Judge-facing paths"
        )
    if not set(packet_paths).issubset(selected_path_set):
        raise ValueError(
            "frozen functional prejudgement group packet paths must be "
            "selected Judge-facing paths"
        )
    all_paths = list(
        dict.fromkeys([*paths, *cross_group_paths, *packet_paths])
    )
    artifact_hashes = value.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError(
            "frozen functional prejudgement artifact hashes are required"
        )
    if set(artifact_hashes) != set(all_paths):
        raise ValueError(
            "frozen functional prejudgement artifact hash coverage mismatch"
        )
    for raw_path in all_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "frozen functional prejudgement artifact does not exist: "
                f"{path}"
            )
        if _content_sha256(path) != str(artifact_hashes[raw_path]):
            raise ValueError(
                "frozen functional prejudgement artifact hash mismatch: "
                f"{raw_path}"
            )
    return FunctionalPrejudgementEvidenceResult(
        status=str(value.get("status") or "frozen"),
        selected_judge_probe_paths=tuple(paths),
        cross_group_probe_paths=tuple(cross_group_paths),
        cross_group_probe_packet=deepcopy(cross_group_packet),
        group_owned_probe_packets=deepcopy(group_packets),
        functional_discovery=deepcopy(
            value.get("functional_discovery")
        ),
        usable_surface_hypotheses=tuple(
            deepcopy(value.get("usable_surface_hypotheses") or [])
        ),
        functional_boundary_evidence=deepcopy(
            value.get("functional_boundary_evidence")
        ),
        acquisition_plan=deepcopy(value.get("acquisition_plan")),
        unscheduled_discovery_items=tuple(
            deepcopy(value.get("unscheduled_discovery_items") or [])
        ),
        telemetry=deepcopy(value.get("telemetry") or {}),
        budget_usage=deepcopy(value.get("budget_usage") or {}),
        source_identity=deepcopy(source_identity),
        artifact_sha256={
            str(key): str(item) for key, item in artifact_hashes.items()
        },
        provenance={
            **deepcopy(provenance),
            "reuse_mode": "frozen",
            "runtime_calls_performed": False,
        },
        runtime_audit=deepcopy(runtime_audit),
    )


def _runtime_result(
    *,
    request: FunctionalPrejudgementEvidenceRequest,
    paths: list[str],
    audit: dict[str, Any],
    source_manifest: dict[str, Any],
) -> FunctionalPrejudgementEvidenceResult:
    selected = tuple(dict.fromkeys(str(path) for path in paths))
    cross_group = tuple(
        dict.fromkeys(
            str(path)
            for path in audit.get("cross_group_evidence_paths") or []
        )
    )
    # Preserve the exact pre-refactor routing rule: typed discovery owns
    # explicit cross/group packets, while older planners expose their
    # undifferentiated probe list to the global Judge.
    if (
        not cross_group
        and not is_functional_discovery_planner_mode(
            audit.get("planner_mode")
        )
    ):
        cross_group = selected
    group_packets = deepcopy(audit.get("group_probe_packets") or {})
    all_artifacts = list(
        dict.fromkeys(
            [
                *selected,
                *cross_group,
                *[
                    str(path)
                    for packet in group_packets.values()
                    if isinstance(packet, dict)
                    for path in packet.get("evidence_paths") or []
                ],
            ]
        )
    )
    detector = source_manifest.get("usable_surface_detector")
    detector = detector if isinstance(detector, dict) else {
        "implementation_id": "not_configured",
        "version": "not_configured",
    }
    provenance = {
        **deepcopy(source_manifest),
        "acquisition_schema_version": (
            FUNCTIONAL_PROBE_ACQUISITION_VERSION
        ),
        "acquisition_plan_version": FUNCTIONAL_ACQUISITION_PLAN_VERSION,
        "usable_surface_detector": deepcopy(detector),
        "runtime_calls_performed": True,
    }
    return FunctionalPrejudgementEvidenceResult(
        status=str(audit.get("status") or "unknown"),
        selected_judge_probe_paths=selected,
        cross_group_probe_paths=cross_group,
        cross_group_probe_packet=functional_probe_judge_packet(
            global_paths=[request.global_image_path],
            probe_paths=list(cross_group),
            acquisition_audit=audit,
        ),
        group_owned_probe_packets=group_packets,
        functional_discovery=deepcopy(
            audit.get("functional_discovery")
        ),
        usable_surface_hypotheses=tuple(
            deepcopy(audit.get("usable_surface_hypotheses") or [])
        ),
        functional_boundary_evidence=deepcopy(
            audit.get("functional_boundary_evidence")
        ),
        acquisition_plan=deepcopy(
            audit.get("functional_acquisition_plan")
        ),
        unscheduled_discovery_items=tuple(
            deepcopy(audit.get("unscheduled_discovery_items") or [])
        ),
        telemetry=_prejudgement_telemetry(audit, selected),
        budget_usage={
            "max_probe_units": int(
                audit.get("max_probe_units")
                if audit.get("max_probe_units") is not None
                else request.max_probe_units
            ),
            "scheduled_probe_count": int(
                audit.get("planned_probe_count")
                or len(audit.get("probe_units") or [])
            ),
            "unscheduled_discovery_count": len(
                audit.get("unscheduled_discovery_items") or []
            ),
            "budget_exhausted": bool(
                audit.get("budget_exhausted")
            ),
            "coverage_complete": bool(
                audit.get("coverage_complete", True)
            ),
        },
        source_identity=request.identity(),
        artifact_sha256={
            path: _content_sha256(Path(path).expanduser().resolve())
            for path in all_artifacts
            if Path(path).expanduser().is_file()
        },
        provenance=provenance,
        runtime_audit=deepcopy(audit),
    )


def _prejudgement_telemetry(
    audit: dict[str, Any],
    selected_paths: tuple[str, ...],
) -> dict[str, Any]:
    probe_results = [
        item
        for item in audit.get("probe_results") or []
        if isinstance(item, dict)
    ]
    usages = [
        item.get("provider_usage")
        for item in probe_results
        if isinstance(item.get("provider_usage"), dict)
    ]
    discovery = audit.get("functional_discovery")
    calls = (
        (discovery.get("provenance") or {}).get("calls")
        if isinstance(discovery, dict)
        and isinstance(discovery.get("provenance"), dict)
        else {}
    )
    planner_calls = (
        len(calls)
        if isinstance(calls, dict) and calls
        else int(
            audit.get("planner_mode") is not None
            and audit.get("status") not in {"not_configured"}
        )
    )
    return {
        "planner_calls": planner_calls,
        "usable_surface_detector_calls": int(
            audit.get("usable_surface_decoder_calls") or 0
        ),
        "selector_calls": sum(
            int(item.get("selector_calls") or 0) for item in usages
        ),
        "preview_render_count": int(
            audit.get("usable_surface_preview_render_count") or 0
        )
        + sum(
            int(item.get("preview_render_count") or 0)
            for item in usages
        ),
        "full_render_count": sum(
            int(item.get("final_render_count") or 0)
            for item in usages
            if item.get("cache_hit") is not True
        ),
        "judge_facing_image_count": len(selected_paths),
        "cache_hits": int(
            audit.get("usable_surface_cache_hits") or 0
        )
        + sum(int(item.get("cache_hit") is True) for item in usages),
    }


def _empty_stage_telemetry() -> dict[str, int]:
    return {
        "planner_calls": 0,
        "usable_surface_detector_calls": 0,
        "selector_calls": 0,
        "preview_render_count": 0,
        "full_render_count": 0,
        "judge_facing_image_count": 0,
        "cache_hits": 0,
    }


def _provider_detector_manifest(provider: Any) -> dict[str, Any]:
    detector = getattr(provider, "usable_surface_detector", None)
    manifest = getattr(detector, "manifest", None)
    if callable(manifest):
        value = manifest()
        if isinstance(value, dict):
            return deepcopy(value)
    policy = getattr(provider, "policy_config", None)
    usable = (
        policy.get("usable_surface")
        if isinstance(policy, dict)
        and isinstance(policy.get("usable_surface"), dict)
        else {}
    )
    return {
        "implementation_id": str(
            usable.get("backend")
            or DEFAULT_USABLE_SURFACE_DETECTOR_BACKEND
        ),
        "version": str(
            usable.get("detector_version")
            or usable.get("prompt_version")
            or "not_configured"
        ),
        "configuration": deepcopy(usable),
    }


def _component_identity(value: Any) -> dict[str, Any]:
    if value is None:
        return {"configured": False}
    return {
        "configured": True,
        "implementation": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
    }


def _validated_path_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"frozen functional prejudgement {label} must be a list"
        )
    result = [
        str(item)
        for item in value
        if isinstance(item, (str, Path)) and str(item).strip()
    ]
    if len(result) != len(value) or len(result) != len(set(result)):
        raise ValueError(
            f"frozen functional prejudgement {label} must contain unique "
            "non-empty paths"
        )
    return result


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
