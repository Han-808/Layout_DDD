"""Strict loader for the path-free generation retrieval catalog v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from ._common import (
    RetrievalContractError,
    array_value,
    canonical_json_bytes,
    exact_keys,
    sha256_bytes,
    sha256_file,
    strict_json_object,
)
from .dataset import DatasetDescriptor
from .encoder import EncoderDescriptor
from .index import IndexDescriptor
from .profile import ComposedRetrievalProfile, RetrievalProfile


CATALOG_SCHEMA_VERSION = "generation_retrieval_catalog_v2"
_T = TypeVar("_T")


def _by_id(values: Iterable[_T], attribute: str, *, label: str) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for value in values:
        key = getattr(value, attribute)
        if key in result:
            raise RetrievalContractError(f"duplicate {label} ID: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class RetrievalCatalog:
    path: Path
    sha256: str
    datasets: dict[str, DatasetDescriptor]
    encoders: dict[str, EncoderDescriptor]
    indexes: dict[str, IndexDescriptor]
    profiles: dict[str, RetrievalProfile]

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalCatalog":
        catalog_path = Path(path).expanduser().resolve()
        if not catalog_path.is_file() or catalog_path.is_symlink():
            raise RetrievalContractError("retrieval catalog must be a regular file")
        raw = strict_json_object(catalog_path)
        exact_keys(
            raw,
            label="retrieval catalog",
            required=(
                "schema_version",
                "dataset_descriptors",
                "encoder_descriptors",
                "index_descriptors",
                "retrieval_profiles",
            ),
        )
        if raw["schema_version"] != CATALOG_SCHEMA_VERSION:
            raise RetrievalContractError(
                f"retrieval catalog schema must be {CATALOG_SCHEMA_VERSION!r}"
            )
        datasets = _by_id(
            (
                DatasetDescriptor.parse(item, label=f"dataset_descriptors[{index}]")
                for index, item in enumerate(
                    array_value(raw["dataset_descriptors"], label="dataset_descriptors")
                )
            ),
            "dataset_id",
            label="dataset",
        )
        encoders = _by_id(
            (
                EncoderDescriptor.parse(item, label=f"encoder_descriptors[{index}]")
                for index, item in enumerate(
                    array_value(raw["encoder_descriptors"], label="encoder_descriptors")
                )
            ),
            "encoder_id",
            label="encoder",
        )
        indexes = _by_id(
            (
                IndexDescriptor.parse(item, label=f"index_descriptors[{index}]")
                for index, item in enumerate(
                    array_value(raw["index_descriptors"], label="index_descriptors")
                )
            ),
            "index_id",
            label="index",
        )
        profiles = _by_id(
            (
                RetrievalProfile.parse(item, label=f"retrieval_profiles[{index}]")
                for index, item in enumerate(
                    array_value(raw["retrieval_profiles"], label="retrieval_profiles")
                )
            ),
            "retrieval_profile_id",
            label="retrieval profile",
        )
        if not datasets or not encoders or not indexes or not profiles:
            raise RetrievalContractError("retrieval catalog descriptor arrays must not be empty")
        catalog = cls(
            path=catalog_path,
            sha256=sha256_file(catalog_path),
            datasets=datasets,
            encoders=encoders,
            indexes=indexes,
            profiles=profiles,
        )
        for profile_id in profiles:
            catalog.compose(profile_id)
        return catalog

    def compose(self, profile_id: str) -> ComposedRetrievalProfile:
        try:
            profile = self.profiles[profile_id]
            dataset = self.datasets[profile.dataset_id]
            encoder = self.encoders[profile.encoder_id]
            index = self.indexes[profile.index_id]
        except KeyError as exc:
            raise RetrievalContractError(
                f"retrieval profile {profile_id!r} references an unknown descriptor: {exc}"
            ) from exc
        profile_value = {
            "retrieval_profile_id": profile.retrieval_profile_id,
            "dataset_id": profile.dataset_id,
            "encoder_id": profile.encoder_id,
            "index_id": profile.index_id,
            "official_strict": profile.official_strict,
            "policy": {
                "algorithm": profile.policy.algorithm,
                "category_argument": profile.policy.category_argument,
                "size_tolerance": profile.policy.size_tolerance,
                "top_k": profile.policy.top_k,
                "min_score": profile.policy.min_score,
                "tie_order": profile.policy.tie_order,
            },
            "golden_suite": {
                "path": profile.golden_suite.path,
                "sha256": profile.golden_suite.sha256,
                "score_tolerance": profile.golden_suite.score_tolerance,
            },
        }
        composed = ComposedRetrievalProfile(
            profile=profile,
            dataset=dataset,
            encoder=encoder,
            index=index,
            catalog_path=self.path,
            catalog_sha256=self.sha256,
            profile_sha256=sha256_bytes(canonical_json_bytes(profile_value)),
        )
        composed.validate_composition()
        return composed
