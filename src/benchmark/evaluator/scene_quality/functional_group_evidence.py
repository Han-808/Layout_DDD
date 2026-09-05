"""Shared visual-evidence bank for atomic Functional group checks.

The bank is routing and audit state only.  It cannot create checks, metric
defects, or verdicts, and it never forwards one check's semantic result to a
later check.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from benchmark.visual_judge.orchestration.evidence_window import (
    EVIDENCE_WINDOW_SCHEMA_VERSION,
    SHARED_GROUP_BANK_POLICY,
    evidence_artifact_id,
)


FUNCTIONAL_GROUP_EVIDENCE_BANK_VERSION = (
    "functional_group_evidence_bank_v1"
)


@dataclass
class FunctionalGroupEvidenceBank:
    group_id: str
    max_active_images: int
    fixed_artifact_ids: list[str] = field(default_factory=list)
    _artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    _events: list[dict[str, Any]] = field(default_factory=list)
    _sequence: int = 0

    @classmethod
    def from_packet(
        cls,
        packet: dict[str, Any],
        *,
        max_active_images: int,
    ) -> "FunctionalGroupEvidenceBank":
        group = packet.get("group")
        group_id = str(
            group.get("group_id") if isinstance(group, dict) else ""
        ).strip()
        if not group_id:
            raise ValueError("Functional group evidence bank requires group_id")
        if (
            isinstance(max_active_images, bool)
            or not isinstance(max_active_images, int)
            or max_active_images < 2
        ):
            raise ValueError(
                "Functional group active evidence window must be >= 2"
            )
        result = cls(
            group_id=group_id,
            max_active_images=max_active_images,
        )
        resolution = packet.get("resolution")
        reuse = (
            resolution.get("functional_probe_reuse")
            if isinstance(resolution, dict)
            and isinstance(
                resolution.get("functional_probe_reuse"), dict
            )
            else {}
        )
        packet_paths = _unique_evidence(packet.get("paths"))
        baseline = _unique_evidence(reuse.get("baseline_packet_paths"))
        if len(baseline) < 2:
            baseline = packet_paths[:2]
        if len(baseline) < 2:
            raise ValueError(
                "shared Functional evidence requires angled global and "
                "group-local fixed evidence"
            )
        fixed = baseline[:2]
        for role, artifact in zip(
            ("angled_global", "group_local"),
            fixed,
            strict=True,
        ):
            artifact_id = result.add_artifact(
                artifact,
                role=role,
                fixed=True,
                source_kind="group_seed",
                source_check_id=None,
                check_ids=[],
                target_ids=_group_member_ids(packet),
                required_observations=[],
                provenance={
                    "source": "baseline_group_packet",
                    "group_id": group_id,
                },
            )
            result.fixed_artifact_ids.append(artifact_id)

        functional = packet.get("functional_probe_evidence")
        image_order = (
            functional.get("image_order")
            if isinstance(functional, dict)
            and isinstance(functional.get("image_order"), list)
            else []
        )
        metadata_by_id: dict[str, dict[str, Any]] = {}
        for item in image_order:
            if not isinstance(item, dict):
                continue
            raw_artifact = item.get("artifact_id")
            if raw_artifact is None:
                continue
            artifact_id = evidence_artifact_id(raw_artifact)
            metadata_by_id[artifact_id] = deepcopy(item)

        requested = _unique_evidence(reuse.get("requested_probe_paths"))
        if not requested:
            requested = [
                item
                for item in packet_paths
                if evidence_artifact_id(item)
                not in set(result.fixed_artifact_ids)
            ]
        for artifact in requested:
            artifact_id = evidence_artifact_id(artifact)
            if artifact_id in set(result.fixed_artifact_ids):
                continue
            metadata = metadata_by_id.get(artifact_id, {})
            result.add_artifact(
                artifact,
                role=str(metadata.get("role") or "functional_probe"),
                fixed=False,
                source_kind="prejudgement_probe",
                source_check_id=None,
                check_ids=_unique_strings(metadata.get("check_ids")),
                target_ids=_unique_strings(
                    [
                        *list(metadata.get("target_ids") or []),
                        *list(metadata.get("related_target_ids") or []),
                    ]
                ),
                required_observations=_unique_strings(
                    metadata.get("required_observations")
                ),
                provenance={
                    "source": "functional_probe_evidence.image_order",
                    "probe_id": metadata.get("probe_id"),
                    "probe_kind": metadata.get("probe_kind"),
                },
            )
        result._events.append(
            {
                "event": "bank_initialized",
                "group_id": group_id,
                "fixed_artifact_ids": list(result.fixed_artifact_ids),
                "reusable_artifact_ids": [
                    artifact_id
                    for artifact_id, record in result._artifacts.items()
                    if not record["fixed"]
                ],
            }
        )
        return result

    def add_artifact(
        self,
        artifact: Any,
        *,
        role: str,
        fixed: bool,
        source_kind: str,
        source_check_id: str | None,
        check_ids: list[str],
        target_ids: list[str],
        required_observations: list[str],
        provenance: dict[str, Any],
    ) -> str:
        artifact_id = evidence_artifact_id(artifact)
        existing = self._artifacts.get(artifact_id)
        source = {
            "source_kind": str(source_kind),
            "source_check_id": (
                str(source_check_id) if source_check_id else None
            ),
            "provenance": deepcopy(provenance),
        }
        if existing is not None:
            if bool(existing["fixed"]) != bool(fixed) and fixed:
                raise ValueError(
                    "a reusable artifact cannot later become fixed evidence"
                )
            existing["visual_evidence"] = deepcopy(artifact)
            existing["check_ids"] = _merge_strings(
                existing.get("check_ids"), check_ids
            )
            existing["target_ids"] = _merge_strings(
                existing.get("target_ids"), target_ids
            )
            existing["required_observations"] = _merge_strings(
                existing.get("required_observations"),
                required_observations,
            )
            if source not in existing["sources"]:
                existing["sources"].append(source)
            return artifact_id
        self._sequence += 1
        self._artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "visual_evidence": deepcopy(artifact),
            "role": str(role),
            "fixed": bool(fixed),
            "sequence": self._sequence,
            "check_ids": _unique_strings(check_ids),
            "target_ids": _unique_strings(target_ids),
            "required_observations": _unique_strings(
                required_observations
            ),
            "sources": [source],
            "consumer_check_ids": [],
        }
        return artifact_id

    def initial_window(
        self,
        check: dict[str, Any],
        *,
        include_reusable: bool = True,
    ) -> tuple[list[Any], dict[str, Any]]:
        check_id = str(check.get("check_id") or "").strip()
        if not check_id:
            raise ValueError("Functional evidence window requires check_id")
        targets = set(_unique_strings(check.get("target_ids")))
        observations = set(
            _unique_strings(check.get("required_observations"))
        )
        ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
        for record in self._artifacts.values():
            if record["fixed"]:
                continue
            if not include_reusable:
                continue
            exact = check_id in set(record.get("check_ids") or [])
            target_overlap = len(
                targets & set(record.get("target_ids") or [])
            )
            observation_overlap = len(
                observations
                & set(record.get("required_observations") or [])
            )
            if not exact and target_overlap == 0:
                continue
            ranked.append(
                (
                    (
                        1 if exact else 0,
                        target_overlap,
                        observation_overlap,
                        -int(record["sequence"]),
                    ),
                    record,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        capacity = self.max_active_images - len(self.fixed_artifact_ids)
        selected_records = [record for _, record in ranked[:capacity]]
        selected_ids = [
            *self.fixed_artifact_ids,
            *(str(record["artifact_id"]) for record in selected_records),
        ]
        for record in selected_records:
            record["consumer_check_ids"] = _merge_strings(
                record.get("consumer_check_ids"), [check_id]
            )
        artifacts = [
            deepcopy(self._artifacts[artifact_id]["visual_evidence"])
            for artifact_id in selected_ids
        ]
        event = {
            "event": "initial_window_built",
            "schema_version": EVIDENCE_WINDOW_SCHEMA_VERSION,
            "policy": SHARED_GROUP_BANK_POLICY,
            "group_id": self.group_id,
            "check_id": check_id,
            "max_active_images": self.max_active_images,
            "fixed_artifact_ids": list(self.fixed_artifact_ids),
            "selected_artifact_ids": selected_ids,
            "selected_reusable_artifact_ids": [
                str(record["artifact_id"]) for record in selected_records
            ],
        }
        self._events.append(deepcopy(event))
        return artifacts, event

    def window_context(
        self,
        *,
        check: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_WINDOW_SCHEMA_VERSION,
            "policy": SHARED_GROUP_BANK_POLICY,
            "group_id": self.group_id,
            "check_id": str(check.get("check_id") or ""),
            "max_active_images": self.max_active_images,
            "fixed_artifact_ids": list(self.fixed_artifact_ids),
            "reusable_artifacts": [
                deepcopy(record)
                for record in self._artifacts.values()
                if not record["fixed"]
            ],
        }

    def absorb_controller_audit(
        self,
        audit_record: dict[str, Any] | None,
        *,
        check: dict[str, Any],
        initial_window: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        check_id = str(check.get("check_id") or "").strip()
        payload = _controller_audit_payload(audit_record)
        window_audit = (
            payload.get("evidence_window")
            if isinstance(payload.get("evidence_window"), dict)
            else {}
        )
        for event in window_audit.get("events") or []:
            if not isinstance(event, dict):
                continue
            for artifact_id in event.get("reused_artifact_ids") or []:
                record = self._artifacts.get(str(artifact_id))
                if record is not None:
                    record["consumer_check_ids"] = _merge_strings(
                        record.get("consumer_check_ids"), [check_id]
                    )

        rendered_ids: list[str] = []
        for event in payload.get("trace") or []:
            if (
                not isinstance(event, dict)
                or event.get("stage") != "render"
                or event.get("status") != "completed"
            ):
                continue
            result = event.get("result")
            evidence = (
                result.get("visual_evidence")
                if isinstance(result, dict)
                and isinstance(result.get("visual_evidence"), list)
                else []
            )
            for artifact in evidence:
                artifact_id = self.add_artifact(
                    artifact,
                    role="check_camera_evidence",
                    fixed=False,
                    source_kind="check_camera_render",
                    source_check_id=check_id,
                    check_ids=[check_id],
                    target_ids=_unique_strings(check.get("target_ids")),
                    required_observations=_unique_strings(
                        check.get("required_observations")
                    ),
                    provenance={
                        "evidence_round": event.get("evidence_round"),
                        "selection_stage": event.get("selection_stage"),
                        "render_provenance": deepcopy(
                            result.get("provenance")
                            if isinstance(result, dict)
                            else None
                        ),
                    },
                )
                rendered_ids.append(artifact_id)
        initial_ids = _unique_strings(
            window_audit.get("initial_artifact_ids")
        )
        if not initial_ids and isinstance(initial_window, dict):
            initial_ids = _unique_strings(
                initial_window.get("selected_artifact_ids")
            )
        final_ids = _unique_strings(window_audit.get("final_artifact_ids"))
        if not final_ids:
            final_ids = list(initial_ids)
        event = {
            "event": "check_episode_absorbed",
            "group_id": self.group_id,
            "check_id": check_id,
            "rendered_artifact_ids": list(dict.fromkeys(rendered_ids)),
            "final_artifact_ids": final_ids,
        }
        self._events.append(deepcopy(event))
        return {
            "policy": SHARED_GROUP_BANK_POLICY,
            "max_active_images": self.max_active_images,
            "fixed_artifact_ids": list(self.fixed_artifact_ids),
            "initial_artifact_ids": initial_ids,
            "final_artifact_ids": final_ids,
            "events": deepcopy(window_audit.get("events") or []),
            "rendered_artifact_ids": list(dict.fromkeys(rendered_ids)),
            "initial_selection_event": deepcopy(initial_window),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FUNCTIONAL_GROUP_EVIDENCE_BANK_VERSION,
            "policy": SHARED_GROUP_BANK_POLICY,
            "decision_authority": "none",
            "group_id": self.group_id,
            "max_active_images": self.max_active_images,
            "fixed_artifact_ids": list(self.fixed_artifact_ids),
            "artifacts": [
                deepcopy(record) for record in self._artifacts.values()
            ],
            "events": deepcopy(self._events),
        }


def _controller_audit_payload(
    record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    audit = record.get("audit")
    return audit if isinstance(audit, dict) else record


def _group_member_ids(packet: dict[str, Any]) -> list[str]:
    group = packet.get("group")
    return _unique_strings(
        group.get("object_ids") if isinstance(group, dict) else []
    )


def _unique_evidence(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[Any] = []
    seen: set[str] = set()
    for item in value:
        artifact_id = evidence_artifact_id(item)
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        result.append(deepcopy(item))
    return result


def _merge_strings(first: Any, second: Any) -> list[str]:
    return _unique_strings([
        *list(first or []),
        *list(second or []),
    ])


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    )
