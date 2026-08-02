from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATALOG_PLACEMENT_CONTRACT_REVISION = "catalog_placement_v1"
MATERIALIZATION_REVISION = "fixed_catalog_materialization_v1"
INSTANCE_REGISTRY_VERSION = "benchmark_instance_registry_v1"
CONSISTENCY_GATE_VERSION = "materialization_consistency_v1"
READINESS_GATE_VERSION = "submission_readiness_v1"


class MaterializationError(ValueError):
    """A generator artifact cannot be materialized into a trusted evaluator scene."""


class ConsistencyError(MaterializationError):
    """Prepared artifacts disagree at a trusted boundary."""


@dataclass(frozen=True)
class MaterializationResult:
    normalized_scene_path: Path
    instance_registry_path: Path
    trusted_render_source_path: Path
    consistency_report_path: Path
    readiness_report_path: Path
    provenance_path: Path
    hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_scene_path": self.normalized_scene_path.as_posix(),
            "instance_registry_path": self.instance_registry_path.as_posix(),
            "trusted_render_source_path": self.trusted_render_source_path.as_posix(),
            "consistency_report_path": self.consistency_report_path.as_posix(),
            "readiness_report_path": self.readiness_report_path.as_posix(),
            "provenance_path": self.provenance_path.as_posix(),
            "hashes": dict(self.hashes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MaterializationResult":
        if not isinstance(value, dict):
            raise TypeError("materialization result must be a JSON object")
        hashes = value.get("hashes")
        if not isinstance(hashes, dict):
            raise TypeError("materialization result hashes must be a JSON object")
        return cls(
            normalized_scene_path=Path(str(value["normalized_scene_path"])).expanduser().resolve(),
            instance_registry_path=Path(str(value["instance_registry_path"])).expanduser().resolve(),
            trusted_render_source_path=Path(
                str(value["trusted_render_source_path"])
            ).expanduser().resolve(),
            consistency_report_path=Path(
                str(value["consistency_report_path"])
            ).expanduser().resolve(),
            readiness_report_path=Path(
                str(value["readiness_report_path"])
            ).expanduser().resolve(),
            provenance_path=Path(str(value["provenance_path"])).expanduser().resolve(),
        hashes={str(key): str(digest) for key, digest in hashes.items()},
    )
